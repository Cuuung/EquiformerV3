"""Self-Conditioned Denoising (SCD) for EquiformerV3 —— v0.

参考 Perez & Gómez-Bombarelli, *Self-Conditioned Denoising for Atomistic
Representation Learning* (2026)。

v0 与论文的差异：论文把条件向量经 AdaNorm 注入每个 block 的 pre-attention
LayerNorm；v0 改用 DeNS 同款的**输入层加法**注入（写进 L=0 通道），因此不需要
改动 `TransBlockV3` / `core_compute` 的签名，可以整套复用 `EquiformerV3DeNS_OC`
的 direct / gradient / compiled 三条前向路径。AdaNorm 版留给 v1。

数据流（仅在训练且该 step 施加了 DeNS 噪声时走双前向）：
    clean pos -> core_compute -> L=0 特征 -> sum-pool -> MLP -> c  [B, C]
    c -> (按图 dropout 成 mask token) -> zero-init 线性 -> 加到噪声前向的输入嵌入
噪声预测头直接复用父类的 `dens_block`。
"""

import torch

from fairchem.core.common.registry import registry

from .equiformer_v3_dens import EquiformerV3DeNS_OC


class SCDCondHead(torch.nn.Module):
    """把 clean 前向的逐原子 L=0 特征压成"每构型一个向量"的信息瓶颈。"""

    def __init__(self, num_channels):
        super().__init__()
        self.pre_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(num_channels),
            torch.nn.Linear(num_channels, num_channels),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(num_channels),
        )
        self.post_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(num_channels),
            torch.nn.Linear(num_channels, num_channels),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(num_channels),
            torch.nn.Linear(num_channels, num_channels),
        )

    def forward(self, x_scalar, batch, num_graphs):
        x = self.pre_proj(x_scalar)
        c = torch.zeros(
            (num_graphs, x.shape[-1]), device=x.device, dtype=x.dtype
        )
        c.index_add_(0, batch, x)
        return self.post_proj(c)


@registry.register_model("equiformer_v3_scd")
class EquiformerV3SCD_OC(EquiformerV3DeNS_OC):
    """
    Args:
        use_scd (bool):         是否启用自条件。False 时行为退化为纯 DeNS。
        use_force_cond (bool):  是否保留 DeNS 的真实力条件。设为 False 即论文形态的
                                纯自条件去噪（此时 `force_embedding` 被移除，避免
                                DDP unused-parameter 报错；从 DeNS ckpt 续训需
                                `strict=False`）。
        scd_p_dropcond (float): 按图丢弃条件向量、替换为可学 mask token 的概率。
        scd_cond_clip (float):  条件向量的数值截断，对齐 SCD 原实现的稳定性处理。
        scd_detach_cond (bool): True 时切断 clean 前向的梯度（省一次 backward，
                                但与论文不一致，论文让梯度回流 clean 前向）。

    其余参数见 `EquiformerV3DeNS_OC`。
    """

    def __init__(
        self,
        use_scd=True,
        use_force_cond=True,
        scd_p_dropcond=0.2,
        scd_cond_clip=100.0,
        scd_detach_cond=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.use_scd = use_scd
        self.use_force_cond = use_force_cond
        self.scd_p_dropcond = scd_p_dropcond
        self.scd_cond_clip = scd_cond_clip
        self.scd_detach_cond = scd_detach_cond

        if not self.use_force_cond:
            self.force_embedding = None

        self.scd_cond_head = SCDCondHead(self.num_channels)
        self.scd_mask_token = torch.nn.Parameter(torch.zeros(1, self.num_channels))
        self.scd_cond_norm = torch.nn.LayerNorm(self.num_channels)
        self.scd_cond_proj = torch.nn.Linear(self.num_channels, self.num_channels)

        self.apply(self._init_weights)
        torch.nn.init.xavier_uniform_(self.scd_mask_token)
        # 零初始化：初始状态下 SCD 分支恒输出 0，与父类 DeNS 逐位一致，
        # 便于从既有 DeNS ckpt 续训。
        torch.nn.init.constant_(self.scd_cond_proj.weight, 0.0)
        torch.nn.init.constant_(self.scd_cond_proj.bias, 0.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_list = super().no_weight_decay()
        no_wd_list.add("scd_mask_token")
        return no_wd_list

    def _scd_clean_cond(self, data, num_graphs):
        """在未加噪结构上跑一次前向，得到每构型的条件向量 [B, C]。

        只借 `data` 的 pos 建图，用完立刻还回原对象，因此对上游已经做过的
        `pos.requires_grad_` / 应变位移不产生副作用。条件向量是旋转平移不变量，
        且不经过 displacement，故对 virial 无贡献 —— DeNS step 本就屏蔽应力。
        """
        pos_clean = (
            data.pos_clean
            if hasattr(data, "pos_clean")
            else data.pos - data.noise_vec
        )
        pos_noisy = data.pos
        data.pos = pos_clean
        try:
            edge_index, edge_distance, edge_distance_vec, _, _, _ = self.generate_graph(
                data,
                enforce_max_neighbors_strictly=self.enforce_max_neighbors_strictly,
                use_pbc_single=self.use_pbc_single,
            )
        finally:
            data.pos = pos_noisy

        # clean 前向不带任何条件，force_embedding 传 0 标量（广播）
        zero_fe = torch.zeros((), dtype=self.dtype, device=self.device)
        x_scalar, _, _, _ = self.core_compute(
            data.atomic_numbers.long(),
            edge_distance,
            edge_distance_vec,
            edge_index,
            data.batch,
            zero_fe,
        )
        return self.scd_cond_head(x_scalar, data.batch, num_graphs)

    def _scd_cond_embedding(self, data):
        """返回加到输入嵌入上的条件项 [N, (lmax+1)^2, C]（只占 L=0 通道）。"""
        num_graphs = len(data.natoms)
        do_self_cond = (
            self.use_scd
            and self.training
            and getattr(data, "denoising_pos_forward", False)
        )

        if do_self_cond:
            c = self._scd_clean_cond(data, num_graphs)
            if self.scd_detach_cond:
                c = c.detach()
            if self.scd_p_dropcond > 0.0:
                keep = (
                    torch.rand(num_graphs, device=c.device) >= self.scd_p_dropcond
                ).to(c.dtype).view(-1, 1)
                c = c * keep + self.scd_mask_token * (1.0 - keep)
        else:
            # 微调 / 推理：无条件路径，单次前向
            c = self.scd_mask_token.expand(num_graphs, -1)

        c = self.scd_cond_norm(c)
        c = c.clamp(min=-self.scd_cond_clip, max=self.scd_cond_clip)
        c = self.scd_cond_proj(c)
        c = c[data.batch]

        cond_embedding = torch.zeros(
            (c.shape[0], (self.lmax + 1) ** 2, self.num_channels),
            device=c.device,
            dtype=c.dtype,
        )
        cond_embedding[:, 0, :] = c
        return cond_embedding

    def _forward_dens_force_encoding(self, data, cond=None):
        """在 DeNS 的输入条件上叠加 SCD 自条件。

        v0 走输入层注入，`cond`（AdaNorm 用的节点级条件）此处未用到。

        三条前向路径（direct / gradient / compiled）都只通过本方法拿
        `force_embedding` 再传进 `core_compute`，所以这里叠加即可，无需改动它们。
        编译路径下 `fe` 本就是 traced region 的显式入参，梯度可回流 clean 前向。
        """
        if self.use_force_cond:
            (
                force_embedding,
                noise_mask_tensor,
                dens_batch_mask_tensor,
                dens_mask_tensor,
            ) = super()._forward_dens_force_encoding(data)
        else:
            (
                _,
                _,
                noise_mask_tensor,
                dens_batch_mask_tensor,
                dens_mask_tensor,
            ) = self._generate_dens_data(data)
            force_embedding = torch.zeros((), dtype=self.dtype, device=self.device)
            noise_mask_tensor = noise_mask_tensor.view(-1, 1)

        force_embedding = force_embedding + self._scd_cond_embedding(data)

        return (
            force_embedding,
            noise_mask_tensor,
            dens_batch_mask_tensor,
            dens_mask_tensor,
        )

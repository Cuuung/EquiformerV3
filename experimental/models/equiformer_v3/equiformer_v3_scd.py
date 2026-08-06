"""Self-Conditioned Denoising (SCD) for EquiformerV3 —— v0 / v1 并存。

参考 Perez & Gómez-Bombarelli, *Self-Conditioned Denoising for Atomistic
Representation Learning* (2026)。

条件向量的注入方式由 `scd_inject` 选择：v0 的 `'input'` 用 DeNS 同款的**输入层
加法**注入（写进 L=0 通道），不改动 `TransBlockV3` / `core_compute` 的签名，可以
整套复用 `EquiformerV3DeNS_OC` 的 direct / gradient / compiled 三条前向路径；
v1 的 `'adanorm'`（默认）则按论文把条件向量经 AdaNorm 注入每个 block 的
pre-attention LayerNorm，`'both'` 两者叠加。见 `_rebuild_blocks_with_adanorm`。

数据流（仅在训练且该 step 施加了 DeNS 噪声时走双前向）：
    clean pos -> core_compute -> L=0 特征 -> sum-pool -> MLP -> c  [B, C]
    c -> (按图 dropout 成 mask token) -> zero-init 线性 -> 加到噪声前向的输入嵌入
噪声预测头直接复用父类的 `dens_block`。
"""

import torch

from fairchem.core.common.registry import registry

from .equiformer_v3_dens import EquiformerV3DeNS_OC
from .transformer_block import TransBlockV3


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
        scd_inject (str):       条件向量的注入方式：'input' 只走 v0 的输入层加法；
                                'adanorm' 只走 v1 的 AdaNorm（blocks 会被重建）；
                                'both' 两者叠加，但只算一次 cond。
        scd_adanorm_targets (tuple): `scd_inject` 含 adanorm 时，哪些 pre-norm
                                换成 AdaNorm，见 `TransBlockV3`。
        scd_adanorm_scope (str): 见 `EquivariantAdaNorm`。
        scd_adanorm_use_node_feat (bool): 见 `EquivariantAdaNorm`。
        scd_adanorm_detach_node_feat (bool): 见 `EquivariantAdaNorm`。默认 True 对齐
                                参考实现；**保守力（`direct_prediction=False`）配置须设
                                False**，否则 autograd 的力不是能量的真实梯度。
        scd_p_dropcond (float): 按图丢弃条件向量、替换为可学 mask token 的概率。
        scd_cond_clip (float):  条件向量的数值截断，对齐 SCD 原实现的稳定性处理。
        scd_detach_cond (bool): True 时切断 clean 前向的梯度（省一次 backward，
                                但与论文不一致，论文让梯度回流 clean 前向）。
        scd_freeze_element_embedding (str): 冻结输入侧元素嵌入的档位，见
                                `_apply_element_embedding_freeze`。
        scd_freeze_mask_token (bool): 是否额外冻结 `scd_mask_token`。
        scd_reg_noise_std (float): clean 前向输入上的正则化噪声标准差（论文附录
                                A.1，sigma~0.005）。默认 0 关闭，见类文档。

    其余参数见 `EquiformerV3DeNS_OC`。
    """

    _FREEZE_LEVELS = ('none', 'sphere', 'sphere_edge', 'all')

    def __init__(
        self,
        use_scd=True,
        use_force_cond=True,
        scd_inject='adanorm',
        scd_adanorm_targets=('attn', 'ffn'),
        scd_adanorm_scope='per_degree',
        scd_adanorm_use_node_feat=True,
        scd_adanorm_detach_node_feat=True,
        scd_p_dropcond=0.2,
        scd_cond_clip=100.0,
        scd_detach_cond=False,
        scd_freeze_element_embedding='none',
        scd_freeze_mask_token=False,
        scd_reg_noise_std=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert scd_inject in ('input', 'adanorm', 'both'), f"unknown scd_inject: {scd_inject}"
        self.scd_inject = scd_inject

        self.use_scd = use_scd
        self.use_force_cond = use_force_cond
        self.scd_p_dropcond = scd_p_dropcond
        self.scd_cond_clip = scd_cond_clip
        self.scd_detach_cond = scd_detach_cond
        self.scd_freeze_element_embedding = scd_freeze_element_embedding
        self.scd_freeze_mask_token = scd_freeze_mask_token
        self.scd_reg_noise_std = scd_reg_noise_std

        if not self.use_force_cond:
            self.force_embedding = None

        # `use_scd=False` 时**不创建**条件生成链路：这些模块在该模式下无人消费，
        # 建出来只会变成 DDP 的 unused parameter（find_unused_parameters=False 下
        # 直接报 reduction 错误），且白占优化器状态与 ckpt 体积。不建则本类结构上
        # 等同于父类 DeNS。
        if self.use_scd:
            self.scd_cond_head = SCDCondHead(self.num_channels)
            self.scd_mask_token = torch.nn.Parameter(torch.zeros(1, self.num_channels))
            self.scd_cond_norm = torch.nn.LayerNorm(self.num_channels)
            self.scd_cond_proj = torch.nn.Linear(self.num_channels, self.num_channels)

            if self.scd_inject in ('adanorm', 'both'):
                self._rebuild_blocks_with_adanorm(
                    targets=tuple(scd_adanorm_targets),
                    scope=scd_adanorm_scope,
                    use_node_feat=scd_adanorm_use_node_feat,
                    detach_node_feat=scd_adanorm_detach_node_feat,
                )

        self.apply(self._init_weights)
        if self.use_scd:
            torch.nn.init.xavier_uniform_(self.scd_mask_token)
            # 零初始化：初始状态下 SCD 分支恒输出 0，与父类 DeNS 逐位一致，
            # 便于从既有 DeNS ckpt 续训。
            #
            # **只在 input 路径需要时才做**。纯 adanorm 模式下恒等性已由
            # `EquivariantAdaNorm` 自己的 fc[-1] 零初始化保证（layer_norm.py:409），
            # 这里再零初始化就是第二道零门，两者串联会**永久锁死**条件通路：
            #   前向 cond_proj -> 0  =>  fc 的输入恒为 0
            #   反向 dL/d fc[-1].weight = delta (x) 输入 = 0  -> fc[-1].weight 卡在 0
            #        而 fc[-1].weight = 0 又切断通往 cond_proj/cond_head 的梯度
            # 只有 fc[-1].bias 能动，于是调制退化成与构型无关的常数。
            # （实测：5 epoch 后 cond_proj.weight 与 fc[3].weight 仍严格为 0，
            #   条件消融 Δ = 0.00%，即条件完全没起作用。）
            # 'both' 保留零初始化是安全的：加法注入那条路第一步就给 cond_proj 真实梯度。
            if self.scd_inject in ('input', 'both'):
                torch.nn.init.constant_(self.scd_cond_proj.weight, 0.0)
                torch.nn.init.constant_(self.scd_cond_proj.bias, 0.0)

        self._apply_element_embedding_freeze()

    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_list = super().no_weight_decay()
        if self.use_scd:
            no_wd_list.add("scd_mask_token")
        return no_wd_list

    def _apply_element_embedding_freeze(self):
        """按档位冻结输入侧元素嵌入。

        论文附录 B：预训练不冻结元素嵌入会让其趋近于零，导致下游不稳定。
        equiv3 的元素身份有三个入口（sphere / edge-degree / 每个 attention
        block），故分四档。**输出头（force/dens/stress block）不在冻结范围内**
        —— 它们是任务头而非输入通道，SSL 预训练时 dens_block 正是被训练的头。
        """
        level = self.scd_freeze_element_embedding
        assert level in self._FREEZE_LEVELS, f"unknown freeze level: {level}"

        if self.scd_freeze_mask_token and self.use_scd:
            self.scd_mask_token.requires_grad_(False)

        if level == 'none':
            return

        self.sphere_embedding.weight.requires_grad_(False)
        if level == 'sphere':
            return

        for emb in (self.edge_degree_embedding.source_embedding,
                    self.edge_degree_embedding.target_embedding):
            if emb is not None:
                emb.weight.requires_grad_(False)
        if level == 'sphere_edge':
            return

        for block in self.blocks:
            for emb in (block.ga.source_embedding, block.ga.target_embedding):
                if emb is not None:
                    emb.weight.requires_grad_(False)

    def _rebuild_blocks_with_adanorm(self, targets, scope, use_node_feat, detach_node_feat):
        """在父类的 block 参数表上做增量，把两处 pre-norm 换成 AdaNorm。

        父类 `__init__` 已建好 `self.blocks`，这里按同一份配置重建并追加
        AdaNorm 相关参数。重建发生在 `self.apply(self._init_weights)` 之前，
        因此新模块同样会被正常初始化。
        """
        new_blocks = torch.nn.ModuleList()
        for i in range(self.num_layers):
            cfg = self._build_block_config(i)
            cfg.update(
                cond_channels=self.num_channels,
                adanorm_targets=targets,
                adanorm_scope=scope,
                adanorm_use_node_feat=use_node_feat,
                adanorm_detach_node_feat=detach_node_feat,
            )
            new_blocks.append(TransBlockV3(**cfg))
        self.blocks = new_blocks

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
        # 论文附录 A.1 的 regularizing noise（sigma ~ 0.005）：只加在未腐蚀视图上。
        # 注意参考实现仅在 noise_in_loader=False 分支施加，其周期材料配置
        # （pretrain_amp20.yaml, noise_in_loader=True）实际未启用，故默认 0。
        # out-of-place：pos_clean 是新张量，不写回 data.pos_clean，不污染 batch。
        if self.scd_reg_noise_std > 0.0 and self.training:
            pos_clean = pos_clean + torch.randn_like(pos_clean) * self.scd_reg_noise_std
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

    def _scd_cond_vector(self, data):
        """节点级条件向量 [N, C]，或 None（未启用 SCD）。"""
        if not self.use_scd:
            return None
        num_graphs = len(data.natoms)
        do_self_cond = self.training and getattr(data, "denoising_pos_forward", False)

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
            c = self.scd_mask_token.expand(num_graphs, -1)
            # 让 scd_cond_head 恒入 autograd 图（grad 为 0 而非 None）。否则非去噪 step 上
            # 它拿不到梯度，DDP find_unused_parameters=False 会在下一步报 reduction 错误。
            # 与 mask_token 的 `c*keep + mask_token*(1-keep)` 同理。
            zero_x = torch.zeros(
                data.batch.shape[0], self.num_channels,
                device=self.scd_mask_token.device, dtype=self.scd_mask_token.dtype,
            )
            c = c + 0.0 * self.scd_cond_head(zero_x, data.batch, num_graphs)

        c = self.scd_cond_norm(c)
        c = c.clamp(min=-self.scd_cond_clip, max=self.scd_cond_clip)
        c = self.scd_cond_proj(c)
        return c[data.batch]

    def _forward_cond(self, data):
        """AdaNorm 用的 cond；inject 不含 adanorm 时返回 None。"""
        if self.scd_inject not in ('adanorm', 'both'):
            return None
        return self._scd_cond_vector(data)

    def _scd_cond_embedding(self, cond_nodes):
        """把节点级条件写进 L=0，得到可加到输入嵌入上的 [N, (lmax+1)^2, C]。"""
        cond_embedding = torch.zeros(
            (cond_nodes.shape[0], (self.lmax + 1) ** 2, self.num_channels),
            device=cond_nodes.device,
            dtype=cond_nodes.dtype,
        )
        cond_embedding[:, 0, :] = cond_nodes
        return cond_embedding

    def _forward_dens_force_encoding(self, data, cond=None):
        """在 DeNS 的输入条件上叠加 SCD 自条件（`scd_inject` 含 input/both 时）。

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

        if self.use_scd and self.scd_inject in ('input', 'both'):
            # adanorm 分支已经算过 cond，直接复用，避免第二次 clean 前向
            cond_nodes = cond if cond is not None else self._scd_cond_vector(data)
            force_embedding = force_embedding + self._scd_cond_embedding(cond_nodes)

        return (
            force_embedding,
            noise_mask_tensor,
            dens_batch_mask_tensor,
            dens_mask_tensor,
        )

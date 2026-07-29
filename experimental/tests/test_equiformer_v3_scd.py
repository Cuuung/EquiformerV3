"""EquiformerV3-SCD (v0) 冒烟测试：结构 / 等变性 / 梯度连通性 / 两条编译路径。

需要 GPU 与项目训练镜像，直接运行：
    python experimental/tests/test_equiformer_v3_scd.py
非 0 退出码即为失败。
"""
import sys
import torch
from torch_geometric.data import Data, Batch

from fairchem.core.common.utils import setup_imports
from fairchem.core.common.registry import registry

setup_imports()

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

MODEL_CFG = dict(
    use_pbc=True,
    use_pbc_single=True,
    otf_graph=True,
    regress_forces=True,
    regress_stress=True,
    direct_prediction=True,
    max_neighbors=20,
    max_radius=5.0,
    num_radial_basis=10,
    max_num_elements=128,
    num_layers=2,
    num_channels=32,
    attn_hidden_channels=16,
    num_heads=4,
    attn_alpha_channels=16,
    attn_value_channels=8,
    ffn_hidden_channels=64,
    norm_type="merge_layer_norm",
    lmax=2,
    mmax=2,
    attn_grid_resolution_list=[14, 8],
    ffn_grid_resolution_list=[14, 14],
    edge_channels=32,
    drop_path_rate=0.0,
    attn_weights_drop=0.0,
    gradient_checkpointing_block_list=[0, 0],
    avg_num_nodes=1,
)

FAILURES = []


def check(name, ok, extra=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {extra}")
    if not ok:
        FAILURES.append(name)


def make_batch(n_per=6, n_sys=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    datas = []
    for _ in range(n_sys):
        cell = torch.eye(3) * 8.0
        pos = torch.rand((n_per, 3), generator=g) * 8.0
        d = Data(
            pos=pos,
            atomic_numbers=torch.randint(1, 30, (n_per,), generator=g).float(),
            cell=cell.unsqueeze(0),
            natoms=torch.tensor([n_per]),
            fixed=torch.zeros(n_per),
            forces=torch.randn((n_per, 3), generator=g) * 0.1,
            energy=torch.tensor([0.0]),
            pbc=torch.tensor([[True, True, True]]),
        )
        datas.append(d)
    return Batch.from_data_list(datas).to(DEV)


def add_noise(batch, std=0.05):
    """复刻 add_gaussian_noise_to_position 的 all_atoms=True 分支。"""
    noise = torch.randn_like(batch.pos) * std
    batch.pos_clean = batch.pos.clone()
    batch.pos = batch.pos + noise
    batch.noise_vec = noise
    batch.denoising_pos_forward = True
    batch.dens_batch_mask = torch.ones(
        (len(batch.natoms),), dtype=torch.bool, device=batch.pos.device
    )
    return batch


def build(**over):
    cfg = dict(MODEL_CFG)
    cfg.update(over)
    return registry.get_model_class("equiformer_v3_scd")(**cfg).to(DEV)


# ---------------------------------------------------------------- 1. 注册
try:
    cls = registry.get_model_class("equiformer_v3_scd")
    check("模型注册 equiformer_v3_scd", cls is not None, f"-> {cls.__name__}")
except Exception as e:
    check("模型注册 equiformer_v3_scd", False, repr(e))
    sys.exit(1)

# scd_inject="input": 本文件里这些 v0 遗留用例（早于 scd_inject 概念）依赖 cond
# 直接加性注入输入嵌入来验证梯度回流；默认值 'adanorm' 下 cond 只经过零初始化
# 的 AdaNorm 末层，未训练时反传到 cond 生成链路的梯度恒为 0（zero-init 门控的
# 数学性质，非 bug），因此显式锁定为 'input' 以保留这些用例的原始语义。
model = build(scd_inject="input")

# ------------------------------------------- 2. 零初始化 => 初始等价于 DeNS
b = add_noise(make_batch())
model.train()
model.dtype, model.device = b.pos.dtype, b.pos.device
model_in = build(scd_inject="input")
model_in.train()
model_in.dtype, model_in.device = b.pos.dtype, b.pos.device
cond = model_in._scd_cond_embedding(model_in._scd_cond_vector(b))
check(
    "零初始化下 cond 恒为 0（初始逐位等价 DeNS，可从 DeNS ckpt 续训）",
    bool(torch.all(cond == 0)),
    f"shape={tuple(cond.shape)}",
)
check(
    "cond 只占 L=0 通道",
    cond.shape[1] == (MODEL_CFG["lmax"] + 1) ** 2,
)

# 打破零初始化，后续测试才有意义
with torch.no_grad():
    model.scd_cond_proj.weight.normal_(0, 0.05)
    model.scd_cond_proj.bias.normal_(0, 0.05)

# ------------------------------------------------------- 3. direct 双前向
b = add_noise(make_batch())
# scd_p_dropcond=0.2 + num_graphs=2：两图同时被丢的概率 4%，此时 scd_cond_head
# 梯度恒零，下面的非零梯度断言会假失败。固定种子消除这个不确定性。
torch.manual_seed(101)
out = model(b)
check(
    "direct + denoising: 前向可跑",
    out["energy"].shape == (2,) and out["forces"].shape == (12, 3),
    f"E{tuple(out['energy'].shape)} F{tuple(out['forces'].shape)} S{tuple(out['stress'].shape)}",
)

def grad_status(m):
    """(全部参数是否进入 autograd 图, 除 mask_token 外是否都拿到非零梯度)

    mask_token 只在 dropcond 命中时才有非零梯度，但 `mask_token * (1 - keep)`
    保证它恒在图中（grad 为 0 而非 None）—— 这是 DDP find_unused_parameters=False
    的判据。
    """
    in_graph, nonzero = {}, {}
    for n, p in m.named_parameters():
        if not n.startswith("scd_"):
            continue
        in_graph[n] = p.grad is not None
        if n != "scd_mask_token":
            nonzero[n] = p.grad is not None and bool(p.grad.abs().sum() > 0)
    return in_graph, nonzero


loss = out["energy"].sum() + out["forces"].sum()
loss.backward()
in_graph, nonzero = grad_status(model)
check(
    "direct: 全部 SCD 参数进入 autograd 图（DDP unused-param 安全）",
    all(in_graph.values()),
    f"{sum(in_graph.values())}/{len(in_graph)}",
)
check(
    "direct: 梯度非零回流（含 clean 前向路径）",
    all(nonzero.values()) and any("cond_head" in k for k, v in nonzero.items() if v),
    f"{sum(nonzero.values())}/{len(nonzero)}",
)
model.zero_grad(set_to_none=True)

# -------------------------------------------- 4. 无噪声 step 走单次前向
b2 = make_batch(seed=1)
n_graph_calls = []
orig_gg = model.generate_graph
model.generate_graph = lambda *a, **k: (n_graph_calls.append(1), orig_gg(*a, **k))[1]
model(b2)
single = len(n_graph_calls)
n_graph_calls.clear()
model(add_noise(make_batch(seed=1)))
double = len(n_graph_calls)
model.generate_graph = orig_gg
check(
    "无噪声 step 单次建图 / 加噪 step 两次建图",
    single == 1 and double == 2,
    f"{single} vs {double}",
)

# ---------------------------------------------------------- 5. eval 单前向
model.eval()
n_graph_calls.clear()
model.generate_graph = lambda *a, **k: (n_graph_calls.append(1), orig_gg(*a, **k))[1]
with torch.no_grad():
    model(add_noise(make_batch(seed=2)))
model.generate_graph = orig_gg
check("eval 态不触发 clean 前向（推理无条件开销）", len(n_graph_calls) == 1, f"{len(n_graph_calls)}")

# ------------------------------------------------------------- 6. 等变性
model.train()


def rot_mat(dtype):
    a = torch.tensor(0.7, dtype=dtype)
    ca, sa = torch.cos(a), torch.sin(a)
    return torch.tensor(
        [[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]], dtype=dtype, device=DEV
    )


torch.manual_seed(3)
b_a = add_noise(make_batch(seed=3))
R = rot_mat(b_a.pos.dtype)
b_b = add_noise(make_batch(seed=3))
b_b.pos = b_a.pos @ R.T
b_b.pos_clean = b_a.pos_clean @ R.T
b_b.noise_vec = b_a.noise_vec @ R.T
b_b.cell = torch.einsum("bij,kj->bik", b_a.cell, R)
b_b.forces = b_a.forces @ R.T

model.eval()  # 关掉 dropcond 随机性
with torch.no_grad():
    o_a = model(b_a)
    o_b = model(b_b)
e_err = (o_a["energy"] - o_b["energy"]).abs().max().item()
f_err = (o_a["forces"] @ R.T - o_b["forces"]).abs().max().item()
check("能量旋转不变", e_err < 1e-4, f"max|dE|={e_err:.2e}")
check("力旋转等变", f_err < 1e-4, f"max|dF|={f_err:.2e}")

# --------------------------------------- 7. use_force_cond=False（纯 SCD）
m2 = build(use_force_cond=False)
has_fe = any(n.startswith("force_embedding") for n, _ in m2.named_parameters())
check("use_force_cond=False 时移除 force_embedding（避免 DDP unused param）", not has_fe)
m2.train()
with torch.no_grad():
    m2.scd_cond_proj.weight.normal_(0, 0.05)
o2 = m2(add_noise(make_batch(seed=4)))
check("纯自条件模式前向可跑", o2["energy"].shape == (2,))

# ------------------------------------------------- 8. 保守力 (gradient) 路径
m3 = build(direct_prediction=False, regress_stress=True, scd_inject="input")
m3.train()
with torch.no_grad():
    m3.scd_cond_proj.weight.normal_(0, 0.05)
b3 = add_noise(make_batch(seed=5))
# 同上：固定种子避免 dropcond 两图同丢导致的假失败。
torch.manual_seed(202)
o3 = m3(b3)
check(
    "gradient(保守力) 路径可跑",
    o3["energy"].shape == (2,) and o3["forces"].shape == (12, 3),
)
o3["energy"].sum().backward()
in_graph3, nonzero3 = grad_status(m3)
check(
    "保守力: 全部 SCD 参数进入 autograd 图（DDP unused-param 安全）",
    all(in_graph3.values()),
    f"{sum(in_graph3.values())}/{len(in_graph3)}",
)
check(
    "保守力: 梯度非零回流（能量二阶图穿过 cond）",
    all(nonzero3.values()),
    f"{sum(nonzero3.values())}/{len(nonzero3)}",
)

# ------------------------------------------------------- 9. no_weight_decay
check("scd_mask_token 进入 no_weight_decay", "scd_mask_token" in model.no_weight_decay())

# ---------------------- 10. 非去噪 step 上 scd_cond_head 恒入图（DDP 安全） ----------------------
# 非去噪 step 走 else 分支（mask_token），若 scd_cond_head 不被强行拉入图，
# 它的 14 个参数 requires_grad=True 但 grad is None，DDP find_unused_parameters=False
# 下一步就会抛 "Expected to have finished reduction..."。loss 必须用
# energy+forces+stress 合成，只用 energy 的话输出头拿不到梯度，会因无关原因失败。
m_nd = build()
m_nd.train()
with torch.no_grad():
    m_nd.scd_cond_proj.weight.normal_(0, 0.05)
b_nd = make_batch(seed=90)  # 不加噪：无 denoising_pos_forward，走 else 分支
o_nd = m_nd(b_nd)
loss_nd = o_nd["energy"].sum() + o_nd["forces"].sum() + o_nd["stress"].sum()
loss_nd.backward()
missing_grad_nd = [
    n for n, p in m_nd.named_parameters() if p.requires_grad and p.grad is None
]
check(
    "非去噪 step: 无任何 requires_grad=True 参数 grad 为 None（DDP find_unused_parameters=False 安全）",
    not missing_grad_nd,
    f"{missing_grad_nd[:5]}",
)

# =================================== 编译路径 ===================================



def run_two_steps(m, tag, seeds=(11, 12)):
    """跑两个不同 batch，验证不会把第一批的 per-batch 张量烤进图（stale-bake）。"""
    outs = []
    for s in seeds:
        b = add_noise(make_batch(n_per=6 + (s % 3), seed=s))
        o = m(b)
        o["energy"].sum().backward()
        m.zero_grad(set_to_none=True)
        outs.append(o["energy"].detach().clone())
    check(f"{tag}: 连续两个不同 shape 的 batch 均可跑（无 stale-bake）", True,
          f"E1={outs[0].tolist()} E2={outs[1].tolist()}")
    return outs


# ---------------------------------------------- A. 外层 torch.compile (direct)
try:
    m = build()
    m.train()
    with torch.no_grad():
        m.scd_cond_proj.weight.normal_(0, 0.05)
    torch._dynamo.config.optimize_ddp = False
    mc = torch.compile(m, dynamic=True)
    run_two_steps(mc, "A/外层 torch.compile + direct")
except Exception as e:
    check("A/外层 torch.compile + direct", False, f"{type(e).__name__}: {str(e)[:300]}")

# ------------------------------------------- B. 内层 make_fx region (保守力)
try:
    torch._dynamo.reset()
    m2 = build(direct_prediction=False, regress_stress=True,
               enable_compile=True, compile_dynamic=False, scd_inject="input")
    m2.train()
    with torch.no_grad():
        m2.scd_cond_proj.weight.normal_(0, 0.05)
    run_two_steps(m2, "B/model.enable_compile + 保守力")

    g = {n: (p.grad is not None) for n, p in m2.named_parameters() if n.startswith("scd_")}
    b = add_noise(make_batch(seed=13))
    # 同上：固定种子避免 dropcond 两图同丢导致的假失败。
    torch.manual_seed(606)
    o = m2(b)
    o["energy"].sum().backward()
    nz = [bool(p.grad is not None and p.grad.abs().sum() > 0)
          for n, p in m2.named_parameters()
          if n.startswith("scd_") and n != "scd_mask_token"]
    check("B: 编译区外的 clean 前向仍收到梯度（fe 作为 traced 入参可微）",
          all(nz), f"{sum(nz)}/{len(nz)}")
except Exception as e:
    check("B/model.enable_compile + 保守力", False, f"{type(e).__name__}: {str(e)[:300]}")

# ------------------------------ C. 编译 vs eager 数值一致性（保守力，同权重）
try:
    torch._dynamo.reset()
    torch.manual_seed(99)
    m3 = build(direct_prediction=False, regress_stress=True)
    with torch.no_grad():
        m3.scd_cond_proj.weight.normal_(0, 0.05)
        # 默认 scd_inject='adanorm'：不打破 AdaNorm 调制头的零初始化，cond 对
        # 输出毫无影响，即便编译区把 cond 整个丢掉本测试也测不出来（假阳性）。
        for blk in m3.blocks:
            if getattr(blk, "use_adanorm_1", False):
                blk.norm_1.fc[-1].weight.normal_(0, 0.05)
                blk.norm_1.fc[-1].bias.normal_(0, 0.05)
                blk.norm_2.fc[-1].weight.normal_(0, 0.05)
                blk.norm_2.fc[-1].bias.normal_(0, 0.05)
    sd = {k: v.clone() for k, v in m3.state_dict().items()}
    m4 = build(direct_prediction=False, regress_stress=True,
               enable_compile=True, compile_dynamic=False)
    m4.load_state_dict(sd)
    # enable_compile 仅在 training 态走编译区；两侧从同一 RNG 状态起跑，
    # 保证 add_noise 的噪声与 dropcond 的丢弃掩码完全一致。
    m3.train(); m4.train()
    torch.manual_seed(7); b1 = add_noise(make_batch(seed=21)); o_eager = m3(b1)
    torch.manual_seed(7); b2 = add_noise(make_batch(seed=21)); o_comp = m4(b2)
    assert torch.equal(b1.pos, b2.pos), "两侧输入不一致，测试无效"
    de = (o_eager["energy"] - o_comp["energy"]).abs().max().item()
    df = (o_eager["forces"] - o_comp["forces"]).abs().max().item()
    check("C: 编译 vs eager 数值一致（保守力）", de < 1e-4 and df < 1e-4,
          f"max|dE|={de:.2e} max|dF|={df:.2e}")
except Exception as e:
    check("C: 编译 vs eager 数值一致", False, f"{type(e).__name__}: {str(e)[:300]}")

# ------------------- D. plain_compile region (enable_compile + direct)
# SCD 自身不引入二次反传，direct 模式走的是 plain_compile(core_compute)，
# 而非保守力那条 make_fx 路径。
try:
    torch._dynamo.reset()
    m5 = build(direct_prediction=True, enable_compile=True, compile_dynamic=False)
    m5.train()
    with torch.no_grad():
        m5.scd_cond_proj.weight.normal_(0, 0.05)
    run_two_steps(m5, "D/model.enable_compile + direct (plain_compile)", seeds=(31, 32))
    check("D: _compiled_core 已建立", m5._compiled_core is not None)

    # clean 前向必须留在 eager，否则会为 clean 图的形状再触发一次编译
    calls = {"n": 0}
    orig_cc = m5.core_compute
    m5.core_compute = lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), orig_cc(*a, **k))[1]
    m5(add_noise(make_batch(seed=33)))
    m5.core_compute = orig_cc
    check("D: clean 前向走 eager core_compute，主前向用 _compiled_core",
          calls["n"] == 1, f"eager 调用={calls['n']}")

    torch._dynamo.reset()
    torch.manual_seed(99)
    m6 = build(direct_prediction=True)
    with torch.no_grad():
        m6.scd_cond_proj.weight.normal_(0, 0.05)
        # 同上：默认 scd_inject='adanorm'，须打破调制头零初始化才能让 cond
        # 透传对数值比较敏感。
        for blk in m6.blocks:
            if getattr(blk, "use_adanorm_1", False):
                blk.norm_1.fc[-1].weight.normal_(0, 0.05)
                blk.norm_1.fc[-1].bias.normal_(0, 0.05)
                blk.norm_2.fc[-1].weight.normal_(0, 0.05)
                blk.norm_2.fc[-1].bias.normal_(0, 0.05)
    sd6 = {k: v.clone() for k, v in m6.state_dict().items()}
    m7 = build(direct_prediction=True, enable_compile=True, compile_dynamic=False)
    m7.load_state_dict(sd6)
    m6.train(); m7.train()
    torch.manual_seed(7); b6 = add_noise(make_batch(seed=41)); o6 = m6(b6)
    torch.manual_seed(7); b7 = add_noise(make_batch(seed=41)); o7 = m7(b7)
    assert torch.equal(b6.pos, b7.pos), "两侧输入不一致，测试无效"
    de = (o6["energy"] - o7["energy"]).abs().max().item()
    df = (o6["forces"] - o7["forces"]).abs().max().item()
    check("D: plain_compile vs eager 数值一致", de < 1e-4 and df < 1e-4,
          f"max|dE|={de:.2e} max|dF|={df:.2e}")
except Exception as e:
    check("D/plain_compile", False, f"{type(e).__name__}: {str(e)[:300]}")

# ========================= cond 透传（Task 3） =========================
import inspect
from fairchem.experimental.models.equiformer_v3.equiformer_v3 import EquiformerV3_OC
from fairchem.experimental.models.equiformer_v3.equiformer_v3_dens import EquiformerV3DeNS_OC

for fn, name in [
    (EquiformerV3_OC._forward_blocks, "_forward_blocks"),
    (EquiformerV3_OC.core_compute, "EquiformerV3_OC.core_compute"),
    (EquiformerV3DeNS_OC.core_compute, "DeNS.core_compute"),
]:
    params = inspect.signature(fn).parameters
    check(f"{name} 有 cond 形参且默认 None",
          "cond" in params and params["cond"].default is None)

check("DeNS 有 _forward_cond 钩子且默认返回 None",
      EquiformerV3DeNS_OC._forward_cond(None, None) is None)

params = inspect.signature(EquiformerV3DeNS_OC._forward_dens_force_encoding).parameters
check("_forward_dens_force_encoding 接受 cond 形参",
      "cond" in params and params["cond"].default is None)

# gradient checkpointing 分支也必须透传 cond
# scd_inject="input"：本断言验证的是"新增 cond 形参没有破坏 checkpointing 路径
# 下的既有 SCD 梯度回流"，而非 AdaNorm 本身的透传（那由下面 Task 4 的用例专门覆盖），
# 走 input 注入才能在未训练模型上观测到非零梯度。
m_ckpt = build(gradient_checkpointing_block_list=[1, 1], scd_inject="input")
m_ckpt.train()
with torch.no_grad():
    m_ckpt.scd_cond_proj.weight.normal_(0, 0.05)
# 同上：固定种子避免 dropcond 两图同丢导致的假失败。
torch.manual_seed(303)
o_ckpt = m_ckpt(add_noise(make_batch(seed=51)))
o_ckpt["energy"].sum().backward()
nz_ckpt = [
    bool(p.grad is not None and p.grad.abs().sum() > 0)
    for n, p in m_ckpt.named_parameters()
    if n.startswith("scd_") and n != "scd_mask_token"
]
check("gradient checkpointing 路径未被 cond 形参破坏（SCD 梯度仍回流）",
      all(nz_ckpt), f"{sum(nz_ckpt)}/{len(nz_ckpt)}")

# ========================= scd_inject（Task 4） =========================
for mode in ("input", "adanorm", "both"):
    mm = build(scd_inject=mode)
    mm.train()
    with torch.no_grad():
        mm.scd_cond_proj.weight.normal_(0, 0.05)
        for blk in mm.blocks:
            if getattr(blk, "use_adanorm_1", False):
                blk.norm_1.fc[-1].weight.normal_(0, 0.05)
                blk.norm_2.fc[-1].weight.normal_(0, 0.05)
    # 同上：固定种子避免 dropcond 两图同丢导致的假失败。
    torch.manual_seed(404)
    o = mm(add_noise(make_batch(seed=60)))
    o["energy"].sum().backward()
    check(f"scd_inject={mode}: direct 前向+反传可跑", o["energy"].shape == (2,))

    if mode in ("adanorm", "both"):
        # 打破 AdaNorm 零初始化后，梯度应能穿过 fc[-1] 回流到条件生成器
        # （scd_cond_proj / scd_cond_head），而不止是"前向+反传不报错"。
        proj_nz = bool(
            mm.scd_cond_proj.weight.grad is not None
            and mm.scd_cond_proj.weight.grad.abs().sum() > 0
        )
        head_nz = all(
            p.grad is not None and bool(p.grad.abs().sum() > 0)
            for p in mm.scd_cond_head.parameters()
        )
        check(f"scd_inject={mode}: 打破 AdaNorm 零初始化后梯度回流到条件生成器",
              proj_nz and head_nz)

    mg = build(scd_inject=mode, direct_prediction=False)
    mg.train()
    og = mg(add_noise(make_batch(seed=61)))
    check(f"scd_inject={mode}: 保守力前向可跑", og["forces"].shape == (12, 3))

m_ada = build(scd_inject="adanorm")
check("scd_inject=adanorm: blocks 建成 AdaNorm",
      all(b.use_adanorm_1 and b.use_adanorm_2 for b in m_ada.blocks))
m_in = build(scd_inject="input")
check("scd_inject=input: blocks 不建 AdaNorm",
      all(not b.use_adanorm_1 and not b.use_adanorm_2 for b in m_in.blocks))

# identity-init：adanorm 模型未训练调制头时等价于 DeNS
torch.manual_seed(77)
m_scd = build(scd_inject="adanorm")
m_dens_cfg = dict(MODEL_CFG)
m_dens = registry.get_model_class("equiformer_v3_dens")(**m_dens_cfg).to(DEV)
m_scd_sd, m_dens_sd = m_scd.state_dict(), m_dens.state_dict()
# EquivariantAdaNorm 把原 norm 包了一层，键名多出 `norm.` 一级：
# blocks.{i}.norm_{1,2}.affine_{weight,bias}/balance_degree_weight 这 12 个键
# 在 m_scd 侧变成 blocks.{i}.norm_{1,2}.norm.同名键。显式断言这个键集合差异，
# 而不是用 `if k in m_dens_sd` 静默过滤掉它们——否则下面的等价性检查即便这
# 12 个键完全没被复制过去，也会因两侧的 affine_weight 恰好是同一份确定性
# 初始化（torch.ones）而"误判"通过。
adanorm_only_keys = {
    f"blocks.{i}.norm_{n}.norm.{k}"
    for i in range(MODEL_CFG["num_layers"])
    for n in (1, 2)
    for k in ("affine_weight", "affine_bias", "balance_degree_weight")
}
check(
    "identity-init: AdaNorm 的 norm.* 重映射键集合与预期一致（12 个）",
    adanorm_only_keys <= (set(m_scd_sd) - set(m_dens_sd)),
    f"缺失: {adanorm_only_keys - (set(m_scd_sd) - set(m_dens_sd))}",
)
shared = {k: v for k, v in m_scd_sd.items() if k in m_dens_sd}
missing, unexpected = m_dens.load_state_dict(shared, strict=False)
m_scd.eval(); m_dens.eval()
b_a = add_noise(make_batch(seed=78))
b_b = add_noise(make_batch(seed=78))
b_b.pos = b_a.pos.clone(); b_b.pos_clean = b_a.pos_clean.clone()
b_b.noise_vec = b_a.noise_vec.clone()
with torch.no_grad():
    o_scd = m_scd(b_a)
    o_dens = m_dens(b_b)
de = (o_scd["energy"] - o_dens["energy"]).abs().max().item()
check("identity-init: adanorm 模型 == 同权重 DeNS", de < 1e-5, f"max|dE|={de:.2e}")

# EquivariantAdaNorm._load_from_state_dict 钩子：既有（AdaNorm 之前）ckpt 的
# 扁平键名 affine_weight/affine_bias/balance_degree_weight 应被重映射到
# norm.同名键，用 strict=True 加载不报错，且加载后的值确实来自"ckpt"而非
# 该层自身的零初始化默认值。用单个 AdaNorm 子模块而非整模型 strict=True，
# 是因为整模型下 SCD 独有的子模块（scd_cond_head 等）与 AdaNorm 本身新增的
# fc.* / l0_mask / 嵌套 norm.expand_index 在任何实现下都不存在于旧 ckpt，
# strict=True 必然因这些与本次要修的键名重映射无关的差异而报错；已用脚本
# 实测核实（见 final-fix-report.md）。
from fairchem.experimental.models.equiformer_v3.layer_norm import EquivariantAdaNorm

adanorm_mod = m_scd.blocks[0].norm_1
full_sd = {k: v.clone() for k, v in adanorm_mod.state_dict().items()}
old_style_sd = dict(full_sd)
for k in ("affine_weight", "affine_bias", "balance_degree_weight"):
    old_style_sd[k] = torch.randn_like(old_style_sd.pop("norm." + k))

fresh_adanorm = EquivariantAdaNorm(
    norm_type=MODEL_CFG["norm_type"], lmax=adanorm_mod.lmax,
    num_channels=adanorm_mod.num_channels, cond_channels=adanorm_mod.cond_channels,
    scope=adanorm_mod.scope, use_node_feat=adanorm_mod.use_node_feat,
).to(DEV)
try:
    fresh_adanorm.load_state_dict(old_style_sd, strict=True)
    hook_loaded_ok = (
        torch.equal(fresh_adanorm.norm.affine_weight, old_style_sd["affine_weight"])
        and torch.equal(fresh_adanorm.norm.affine_bias, old_style_sd["affine_bias"])
        and torch.equal(fresh_adanorm.norm.balance_degree_weight, old_style_sd["balance_degree_weight"])
        and not torch.equal(fresh_adanorm.norm.affine_weight, full_sd["norm.affine_weight"])
    )
    check("AdaNorm._load_from_state_dict 钩子: 旧 ckpt 扁平键 strict=True 加载成功且值来自 ckpt",
          hook_loaded_ok)
except Exception as e:
    check("AdaNorm._load_from_state_dict 钩子: 旧 ckpt 扁平键 strict=True 加载成功且值来自 ckpt",
          False, f"{type(e).__name__}: {str(e)[:300]}")

# adanorm 模式的旋转等变（调制头非零）
m_eq = build(scd_inject="adanorm")
with torch.no_grad():
    m_eq.scd_cond_proj.weight.normal_(0, 0.05)
    for blk in m_eq.blocks:
        blk.norm_1.fc[-1].weight.normal_(0, 0.05)
        blk.norm_1.fc[-1].bias.normal_(0, 0.05)
        blk.norm_2.fc[-1].weight.normal_(0, 0.05)
m_eq.eval()
ba = add_noise(make_batch(seed=79))
R = rot_mat(ba.pos.dtype)
bb = add_noise(make_batch(seed=79))
bb.pos = ba.pos @ R.T
bb.pos_clean = ba.pos_clean @ R.T
bb.noise_vec = ba.noise_vec @ R.T
bb.cell = torch.einsum("bij,kj->bik", ba.cell, R)
bb.forces = ba.forces @ R.T
with torch.no_grad():
    oa = m_eq(ba)
    ob = m_eq(bb)
e_err = (oa["energy"] - ob["energy"]).abs().max().item()
f_err = (oa["forces"] @ R.T - ob["forces"]).abs().max().item()
check("adanorm: 能量旋转不变", e_err < 1e-4, f"max|dE|={e_err:.2e}")
check("adanorm: 力旋转等变", f_err < 1e-4, f"max|dF|={f_err:.2e}")

# ===================== 元素嵌入冻结四档（Task 5） =====================
# MODEL_CFG: num_channels=32, edge_channels=32, max_num_elements=128, num_layers=2
#   sphere        = 128*32                     = 4096
#   edge_degree   = 2*128*32                   = 8192   -> 累计 12288
#   blocks(2 层)  = 2*2*128*32                 = 16384  -> 累计 28672
EXPECTED_FROZEN = {"none": 0, "sphere": 4096, "sphere_edge": 12288, "all": 28672}

for level, expected in EXPECTED_FROZEN.items():
    mf = build(scd_freeze_element_embedding=level)
    frozen = sum(p.numel() for p in mf.parameters() if not p.requires_grad)
    check(f"冻结档 {level}: 冻结参数数 == {expected}", frozen == expected, f"实际 {frozen}")

# 身份级检查：edge_degree_embedding / blocks[i].ga / force_block / dens_block 的
# source/target_embedding 形状完全相同（同一份 edge_channels_list），仅凭上面的
# 冻结总数无法区分"冻对了模块"还是"冻错了模块但数量凑巧相等"。force_block /
# dens_block 是输出头，不在冻结范围内，all 档下必须仍可训练。
# （stress_block 是 FeedForwardNetworkStressHead，没有 source/target_embedding，不检查。）
mf_all = build(scd_freeze_element_embedding="all")
for head_name in ("force_block", "dens_block"):
    head = getattr(mf_all, head_name)
    for emb_name in ("source_embedding", "target_embedding"):
        emb = getattr(head, emb_name)
        if emb is None:
            check(f"{head_name}.{emb_name} 为 None（use_atom_edge_embedding=False，符合预期）", True)
        else:
            check(f"all 档下 {head_name}.{emb_name} 仍可训练（输出头不在冻结范围）",
                  emb.weight.requires_grad)

# 冻结后仍能前向+反传，且所有 requires_grad=True 的参数都进图
# 注：MODEL_CFG 是 direct_prediction，force_block/dens_block/stress_block 是与
# energy 分支并列的独立输出头，只从各自的 loss 项拿梯度，故须用 energy+forces+
# stress 三项合成 loss（对齐真实训练的多任务 loss），否则即便不冻结任何东西
# 这些头也不会进图，这不是冻结逻辑的问题。
mf = build(scd_freeze_element_embedding="all")
mf.train()
with torch.no_grad():
    mf.scd_cond_proj.weight.normal_(0, 0.05)
of = mf(add_noise(make_batch(seed=70)))
loss = of["energy"].sum() + of["forces"].sum() + of["stress"].sum()
loss.backward()
no_grad_names = [
    n for n, p in mf.named_parameters() if p.requires_grad and p.grad is None
]
check("冻结 all 后 DDP unused-param 安全", not no_grad_names, f"{no_grad_names[:3]}")

mt = build(scd_freeze_mask_token=True)
check("scd_freeze_mask_token=True 冻住 mask token",
      not mt.scd_mask_token.requires_grad)

try:
    build(scd_freeze_element_embedding="bogus")
    check("非法档位应报错", False, "未抛异常")
except AssertionError:
    check("非法档位应报错", True)

# ================= clean 前向正则化噪声（Task 6） =================
# 默认 0 时 clean 坐标必须保持不变（reg noise 是 out-of-place 的，不污染 batch）
m0 = build(scd_reg_noise_std=0.0)
m0.train()
b0 = add_noise(make_batch(seed=81))
before = b0.pos_clean.clone()
m0(b0)
check("scd_reg_noise_std=0: pos_clean 不被修改",
      torch.equal(b0.pos_clean, before))

# reg noise 开启时 pos_clean 同样不该被就地改写
mr0 = build(scd_reg_noise_std=0.05)
mr0.train()
br0 = add_noise(make_batch(seed=83))
before_r = br0.pos_clean.clone()
mr0(br0)
check("scd_reg_noise_std>0: pos_clean 仍不被就地改写",
      torch.equal(br0.pos_clean, before_r))

# 关掉 dropcond 以隔离变量：条件向量的差异必须只来自 reg noise
bq = add_noise(make_batch(seed=82))
m_on = build(scd_reg_noise_std=0.05, scd_p_dropcond=0.0)
m_off = build(scd_reg_noise_std=0.0, scd_p_dropcond=0.0)
m_on.train()
m_off.train()
# 直接调 _scd_cond_vector（不经 forward()）时 self.dtype/self.device 尚未被
# forward() 的开头两行设置，须手动补上，对齐上面 model.dtype/model.device 的用法。
m_on.dtype, m_on.device = bq.pos.dtype, bq.pos.device
m_off.dtype, m_off.device = bq.pos.dtype, bq.pos.device
# scd_cond_proj 零初始化会让 cond 恒为 0，掩盖 reg noise 的影响，须先打破
# （同上文"打破零初始化"用法），否则本组断言无论实现对错都会通过。
with torch.no_grad():
    m_on.scd_cond_proj.weight.normal_(0, 0.05)
    m_off.scd_cond_proj.weight.normal_(0, 0.05)

torch.manual_seed(1)
a1 = m_on._scd_cond_vector(bq)
torch.manual_seed(2)
a2 = m_on._scd_cond_vector(bq)
check("reg noise 开: 两次 clean 前向的条件不同",
      not torch.allclose(a1, a2, atol=1e-6))

torch.manual_seed(1)
c1 = m_off._scd_cond_vector(bq)
torch.manual_seed(2)
c2 = m_off._scd_cond_vector(bq)
check("对照组 reg noise 关: 两次 clean 前向的条件相同",
      torch.allclose(c1, c2, atol=1e-6))

# ============== use_scd=False 退回纯 DeNS（结构与 DDP 安全） ==============
# 语义是「退回纯 DeNS」，所以条件生成链路根本不该被创建 —— 建出来无人消费，
# 在 DDP find_unused_parameters=False 下会直接报 reduction 错误。
m_off = build(use_scd=False)
scd_params = [n for n, _ in m_off.named_parameters() if n.startswith("scd_")]
check("use_scd=False: 不创建任何 scd_* 参数", not scd_params, f"{scd_params[:3]}")
check("use_scd=False: blocks 不含 AdaNorm",
      all(not getattr(b, "use_adanorm_1", False) and not getattr(b, "use_adanorm_2", False)
          for b in m_off.blocks))
check("use_scd=False: no_weight_decay 不含 scd_mask_token",
      "scd_mask_token" not in m_off.no_weight_decay())

# 与父类 DeNS 的参数名集合完全一致 —— 这是「结构上等同」的可执行判据
m_dens_ref = registry.get_model_class("equiformer_v3_dens")(**MODEL_CFG).to(DEV)
off_keys = set(m_off.state_dict().keys())
dens_keys = set(m_dens_ref.state_dict().keys())
check("use_scd=False: state_dict 键集合与 equiformer_v3_dens 完全一致",
      off_keys == dens_keys,
      f"多 {len(off_keys - dens_keys)} 少 {len(dens_keys - off_keys)}")

# 非去噪 step 反传后不得有 requires_grad=True 却 grad is None 的参数
m_off.train()
o_off = m_off(make_batch(seed=91))
(o_off["energy"].sum() + o_off["forces"].sum() + o_off["stress"].sum()).backward()
dead_off = [n for n, p in m_off.named_parameters() if p.requires_grad and p.grad is None]
check("use_scd=False: 非去噪 step 无 DDP unused param", not dead_off, f"{dead_off[:3]}")

# scd_freeze_mask_token=True 在 use_scd=False 下不得抛异常（mask token 不存在）
try:
    build(use_scd=False, scd_freeze_mask_token=True)
    check("use_scd=False + scd_freeze_mask_token=True 不报错", True)
except Exception as e:
    check("use_scd=False + scd_freeze_mask_token=True 不报错", False, repr(e))

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} -> {FAILURES}")
    sys.exit(1)
print("ALL PASS")

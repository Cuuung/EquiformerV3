"""元素嵌入范数诊断埋点的单元测试。"""
import sys
import torch

from fairchem.core.common.utils import setup_imports
from fairchem.core.common.registry import registry

setup_imports()

from fairchem.experimental.trainers.equiformer_v3_dens_trainer import (
    element_embedding_drift,
    element_embedding_norms,
    element_embedding_snapshot,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FAILURES = []


def check(name, ok, extra=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {extra}")
    if not ok:
        FAILURES.append(name)


MODEL_CFG = dict(
    use_pbc=True, use_pbc_single=True, otf_graph=True,
    regress_forces=True, regress_stress=True, direct_prediction=True,
    max_neighbors=20, max_radius=5.0, num_radial_basis=10, max_num_elements=128,
    num_layers=2, num_channels=32, attn_hidden_channels=16, num_heads=4,
    attn_alpha_channels=16, attn_value_channels=8, ffn_hidden_channels=64,
    norm_type="merge_layer_norm", lmax=2, mmax=2,
    attn_grid_resolution_list=[14, 8], ffn_grid_resolution_list=[14, 14],
    edge_channels=32, drop_path_rate=0.0, attn_weights_drop=0.0,
    gradient_checkpointing_block_list=[0, 0], avg_num_nodes=1,
)

m = registry.get_model_class("equiformer_v3_scd")(**MODEL_CFG).to(DEV)
norms = element_embedding_norms(m)

check("返回三个键", set(norms) == {"emb_norm_sphere", "emb_norm_edge_degree", "emb_norm_blocks"},
      f"{sorted(norms)}")
check("全部为有限正数", all(0.0 < v < float("inf") for v in norms.values()), f"{norms}")

with torch.no_grad():
    m.sphere_embedding.weight.mul_(0.0)
norms2 = element_embedding_norms(m)
check("置零后 sphere 范数为 0", norms2["emb_norm_sphere"] == 0.0)
check("置零 sphere 不影响其余两项",
      norms2["emb_norm_edge_degree"] == norms["emb_norm_edge_degree"])

# 对不带 SCD 的普通 DeNS 模型也要能用
md = registry.get_model_class("equiformer_v3_dens")(**MODEL_CFG).to(DEV)
check("对 equiformer_v3_dens 同样可用", len(element_embedding_norms(md)) == 3)

# 敏感性/排除性核对：确保 emb_norm_blocks 统计的是 m.blocks[i].ga 而非
# force_block（输出头，同为 EquivariantGraphAttention，若误统计不会报错，
# 只会静默算错数值）。
assert m.force_block.source_embedding is not None
norms3 = element_embedding_norms(m)
with torch.no_grad():
    m.force_block.source_embedding.weight.mul_(0.0)
    m.force_block.target_embedding.weight.mul_(0.0)
norms4 = element_embedding_norms(m)
check("置零 force_block（输出头）不影响 emb_norm_blocks",
      norms4["emb_norm_blocks"] == norms3["emb_norm_blocks"])

with torch.no_grad():
    m.blocks[0].ga.source_embedding.weight.mul_(0.0)
norms5 = element_embedding_norms(m)
check("置零某 block.ga.source_embedding 会改变 emb_norm_blocks（证明确实统计了该表）",
      norms5["emb_norm_blocks"] != norms4["emb_norm_blocks"])


# ---- 位移范数 ‖W - W_ref‖_F ----
m2 = registry.get_model_class("equiformer_v3_scd")(**MODEL_CFG).to(DEV)
ref = element_embedding_snapshot(m2)
n_before = element_embedding_norms(m2)["emb_norm_sphere"]

d0 = element_embedding_drift(m2, ref)
check("返回三个键", set(d0) == {"emb_drift_sphere", "emb_drift_edge_degree", "emb_drift_blocks"},
      f"{sorted(d0)}")
check("刚取基线时位移恒为 0", all(v == 0.0 for v in d0.values()), f"{d0}")

# 基线必须是副本而非视图：改权重不应连带改基线（否则位移恒为 0，指标失效）
with torch.no_grad():
    m2.sphere_embedding.weight[3].add_(1.0)
check("基线是副本（改权重后位移非零）", element_embedding_drift(m2, ref)["emb_drift_sphere"] > 0.0)

# 核心诉求：只动一行时，位移只反映该行，"死行"贡献恒为 0。
# 期望值 = ‖[1.0] * num_channels‖ = sqrt(C)。
expected = MODEL_CFG["num_channels"] ** 0.5
check("单行位移等于该行的位移范数（死行贡献为 0）",
      abs(element_embedding_drift(m2, ref)["emb_drift_sphere"] - expected) < 1e-4,
      f'{element_embedding_drift(m2, ref)["emb_drift_sphere"]:.4f} vs {expected:.4f}')

# 判别力对比（同一模型的前后自比）：同样这一行改动，整表范数的相对变化不到 1%，
# 而位移指标直接给出 sqrt(C)。
check("整表范数确实被死行稀释（相对变化 < 1%，位移指标则直接给出 sqrt(C)）",
      abs(element_embedding_norms(m2)["emb_norm_sphere"] - n_before) / n_before < 0.01,
      f'{n_before:.4f} -> {element_embedding_norms(m2)["emb_norm_sphere"]:.4f}')

# 塌缩语义：权重归零时位移趋向 ‖W_ref‖ 而非趋向 0
with torch.no_grad():
    m2.sphere_embedding.weight.zero_()
check("塌缩到零时位移等于 ‖W_ref‖",
      abs(element_embedding_drift(m2, ref)["emb_drift_sphere"]
          - float(ref["sphere"][0].norm())) < 1e-3)

# 分组不串台（双向、覆盖两张表）：任一组误收另一组的表都不会报错，只会静默算错，
# 所以每组的 source 和 target 都必须动到。
with torch.no_grad():
    for b in m2.blocks:
        b.ga.source_embedding.weight.add_(1.0)
        b.ga.target_embedding.weight.add_(1.0)
d1 = element_embedding_drift(m2, ref)
check("动遍 block.ga 的 source+target，edge_degree 项仍为 0", d1["emb_drift_edge_degree"] == 0.0)
check("动遍 block.ga 会抬高 blocks 项", d1["emb_drift_blocks"] > 0.0)

with torch.no_grad():
    m2.edge_degree_embedding.source_embedding.weight.add_(1.0)
    m2.edge_degree_embedding.target_embedding.weight.add_(1.0)
d2 = element_embedding_drift(m2, ref)
check("动遍 edge_degree 的 source+target，blocks 项不变",
      d2["emb_drift_blocks"] == d1["emb_drift_blocks"])
check("动遍 edge_degree 会抬高 edge_degree 项", d2["emb_drift_edge_degree"] > 0.0)

check("对 equiformer_v3_dens 同样可用",
      len(element_embedding_drift(md, element_embedding_snapshot(md))) == 3)

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} -> {FAILURES}")
    sys.exit(1)
print("ALL PASS")

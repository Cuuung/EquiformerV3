"""元素嵌入范数诊断埋点的单元测试。"""
import sys
import torch

from fairchem.core.common.utils import setup_imports
from fairchem.core.common.registry import registry

setup_imports()

from fairchem.experimental.trainers.equiformer_v3_dens_trainer import (
    element_embedding_norms,
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

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} -> {FAILURES}")
    sys.exit(1)
print("ALL PASS")

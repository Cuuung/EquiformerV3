"""SCD v1 配置的实例化与关键开关校验。"""
import copy
import sys
import yaml
import torch

from fairchem.core.common.utils import setup_imports
from fairchem.core.common.registry import registry

setup_imports()

P = "experimental/configs/omat24/mptrj/experiments/scd/eager_fp32_N2L2C64.yml"
FAILURES = []


def check(name, ok, extra=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {extra}")
    if not ok:
        FAILURES.append(name)


cfg = yaml.safe_load(open(P))
mc = copy.deepcopy(cfg["model"])
name = mc.pop("name")
m = registry.get_model_class(name)(**mc)

check("模型可实例化", name == "equiformer_v3_scd", f"{m.num_params/1e6:.2f}M")
check("scd_inject=adanorm", m.scd_inject == "adanorm")
check("blocks 全部建成 AdaNorm",
      all(b.use_adanorm_1 and b.use_adanorm_2 for b in m.blocks))
check("eager: optim.use_compile 为 False", cfg["optim"]["use_compile"] is False)
check("eager: model 未设 enable_compile", "enable_compile" not in cfg["model"])
check("drop_path_rate == 0.1", cfg["model"]["drop_path_rate"] == 0.1)
check("冻结档默认 none", m.scd_freeze_element_embedding == "none")
check("reg noise 默认 0.0", m.scd_reg_noise_std == 0.0)
check("N2L2C64 结构",
      cfg["model"]["num_layers"] == 2 and cfg["model"]["lmax"] == 2
      and cfg["model"]["num_channels"] == 64)
check("trainer 为 equiformer_v3_dens_trainer",
      cfg["trainer"] == "equiformer_v3_dens_trainer")

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} -> {FAILURES}")
    sys.exit(1)
print("ALL PASS")

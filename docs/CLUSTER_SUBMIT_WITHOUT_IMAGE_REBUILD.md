# 在"不含 Muon 的旧镜像"上一行命令跑训练(不重建镜像)

> 目标读者:负责 **eSEN** 项目的 Claude Code。
> 场景:集群(SenseCore)给的 docker 镜像里**烤死了一份旧的框架包**(没有你新写的 Muon/HybridMuon 代码)。
> 结论:**不用重建镜像。** 把你挂载到共享盘上的源码用 `PYTHONPATH` 顶到镜像自带包的前面,整个任务就是**一条 `bash <脚本>`**。
> **从 0 写脚本**:直接用 §7 的完整自包含脚本(只改顶部 5 个变量)+ §8 的发现/自检步骤——本文档自给自足,不依赖读本项目仓库的任何文件。

---

## 0. TL;DR

1. 镜像里的包是旧的 → 直接 `optimizer: HybridMuon` 会报 `AttributeError`(镜像里没这个类)。
2. **修法不是重建镜像,而是 `export PYTHONPATH=<你的源码>/src:$PYTHONPATH`** —— Python 搜索 `PYTHONPATH` 在 `site-packages` **之前**,于是 `import <pkg>` 命中你挂载的新源码,镜像里的旧包被"遮蔽"。
3. **绝不在任务里 `pip install`**(没网/会和镜像打架/慢)。只靠 PYTHONPATH。
4. 一行提交命令 = `bash <提交脚本>`;脚本里固定 6 段(见 §3),其中**自检**那段保证你跑的确实是新代码。
5. 共享盘 `/mnt/afs` 在所有计算节点自动挂载 → 源码、config、checkpoint 都放这里,多机可见。

---

## 1. 背景:为什么镜像里"没有 Muon"

集群任务跑在固定 docker 镜像里,镜像在打包时 `pip install` 了一份框架(本项目是 `fairchem`)。你后来在仓库 `src/` 里新增的代码(HybridMuon 优化器等)**不在镜像里**。任务一启动 `import` 到的是镜像 `site-packages` 里的旧包 → 新功能直接 `AttributeError`。

重建+推镜像很重(skopeo 打包、推 registry、几十分钟起步)。完全没必要 —— 用 PYTHONPATH 顶一下即可。

## 2. 核心方案:PYTHONPATH 覆盖(零重建)

```bash
export PYTHONPATH=/mnt/afs/<你的仓库>/src:${PYTHONPATH:-}
```

原理:Python 的 `sys.path` 顺序是 `PYTHONPATH` 条目排在 `site-packages` **前面**,所以 `import <pkg>` 优先解析到你挂载的源码目录。`<你的仓库>/src` 必须是**包含顶层包目录的那一层**(即 `src/<pkg>/__init__.py` 存在)。

> ⚠️ 注意点:
> - 这招只可靠地覆盖**纯 Python 文件**。若 eSEN 有**编译型扩展(CUDA/C++ .so)**烤在镜像里,挂载源码里的 `.py` 能覆盖,但 ABI 敏感的已编译算子要和镜像版本对齐——别用挂载源码去改编译算子的接口,否则可能 import 冲突或段错误。
> - 若镜像用 `pip install -e`(editable)装的包,PYTHONPATH 仍在它前面,通常没问题;**以 §3 的自检打印的 `__file__` 为准**。

## 3. 一行提交命令 + 脚本里必须有的 6 段

**提交命令(就这一行)**:
```bash
bash '/mnt/afs/<你的仓库>/path/to/<提交脚本>.sh'
```

脚本本体固定包含下面 6 段(本项目实例的精简版,eSEN 改占位符即可):

```bash
#!/usr/bin/env bash
set -euo pipefail

# (1) 进仓库根目录(相对路径全都从这里算)
cd /mnt/afs/<你的仓库>

# (2) 清掉从登录机泄漏进来的代理 —— 否则 wandb/外网数据面在计算节点上连不上
#     (127.0.0.1:<port> 那种 loopback 隧道只在登录机存在,计算节点上是死的)
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

# (3) 关键:把挂载源码顶到镜像旧包前面(不要 pip install)
export PYTHONPATH=/mnt/afs/<你的仓库>/src:${PYTHONPATH:-}

# (4) 自检:确认 import 到的确实是"带新代码"的那份,否则立刻报错退出
#     —— 把 'HybridMuon' 换成 eSEN 新代码里的一个标志性符号/字符串
python -c "import inspect, <pkg>.<模块> as m; \
  assert 'HybridMuon' in inspect.getsource(m), 'WRONG pkg imported (no new code): '+m.__file__; \
  print('[submit] pkg OK:', m.__file__)"

# (5) 分布式启动:GPU 数 / 多机 rendezvous 全部从 SenseCore 注入的环境变量取
torchrun \
  --nproc_per_node "${SENSECORE_ACCELERATE_DEVICE_COUNT:-8}" \
  --nnodes        "${SENSECORE_PYTORCH_NNODES:-1}" \
  --node_rank     "${SENSECORE_PYTORCH_NODE_RANK:-0}" \
  --master_addr   "${MASTER_ADDR:-127.0.0.1}" \
  --master_port   "${MASTER_PORT:-29500}" \
  <你的训练入口>.py \
    --config <你的config> \
    --run-dir /mnt/afs/<共享checkpoint目录> \
    --identifier <本次实验名> \
    <其它项目特定参数...>
```

逐段说明:
- **(1) cd 仓库**:让脚本无论从哪触发都用同一套相对路径找 config。
- **(2) unset 代理**:登录机 `~/.bashrc` 常 `export http(s)_proxy / all_proxy` 指向 loopback 隧道;这些变量会被提交 shell 带进任务,在计算节点上是死地址 → wandb 面板全空/超时。**必须清。**
- **(3) PYTHONPATH**:本方案的核心,见 §2。
- **(4) 自检**:最省事的"我到底跑的是不是新代码"探针。失败立刻 `assert` 退出,避免白跑几小时才发现 import 错。**强烈建议保留。**
- **(5) torchrun**:GPU 数从 `SENSECORE_ACCELERATE_DEVICE_COUNT` 取,多机用 `SENSECORE_PYTORCH_NNODES / NODE_RANK + MASTER_ADDR/PORT` 做 rendezvous。`:-` 默认值让同一脚本也能在单机 `bash` 直接跑。
- **(6,可选) 串联多阶段**:本项目脚本里 stage-1 用固定 `--identifier` 跑完,再 glob 出 `best_checkpoint.pt` 喂给 stage-2(`--load_pretrained_weights=<ckpt>`)。checkpoint 落在共享盘 → 所有节点可见。eSEN 若是单阶段就不用这段。

## 4. SenseCore Web UI 侧要做的

- **设备数**:在任务表单里设 GPU 数(脚本里 `SENSECORE_ACCELERATE_DEVICE_COUNT` 会拿到它)。多机再设节点数。
- **启动命令**:就填那一行 `bash '/mnt/afs/.../<脚本>.sh'`。
- **镜像**:选你们现成的(旧框架)镜像,**不用动**。
- **挂载**:`/mnt/afs` 默认自动挂载到每个节点,源码/config/checkpoint 都放这下面即可,无需额外配置。
- **环境变量**:`SENSECORE_*` / `MASTER_ADDR` / `MASTER_PORT` 由平台注入,脚本直接读。

## 5. eSEN 适配清单(把这些占位符换成 eSEN 的)

- [ ] `/mnt/afs/<你的仓库>` → eSEN 仓库在共享盘的实际路径。
- [ ] `PYTHONPATH=.../src` → eSEN 源码里"包含顶层包"的那一层(确认 `src/<pkg>/__init__.py` 在)。
- [ ] 自检里的 `<pkg>.<模块>` 和标志字符串 `'HybridMuon'` → eSEN 新代码里的真实模块 + 标志符号。
- [ ] `<你的训练入口>.py` + 所有 `--config/--run-dir/--identifier/...` → eSEN 自己的 CLI(本项目这些是 fairchem 专有,别照抄)。
- [ ] 共享 checkpoint 目录 → eSEN 自己的。
- [ ] 如有 `num_workers` 之类与 lmdb/数据加载 fork 安全相关的坑,按 eSEN 数据栈设置(本项目用 `--optim.num_workers=0`,因为 lmdb env 不是 fork-safe)。

## 6. 常见坑(排错)

| 症状 | 原因 | 处理 |
|---|---|---|
| `AttributeError: ... HybridMuon` / 找不到新类 | import 到了镜像旧包 | 检查 (3) PYTHONPATH 路径对不对;看 (4) 自检打印的 `__file__` 是不是指向你挂载的 src |
| wandb 面板全空 / "unexpected EOF" / Client.Timeout | 登录机代理变量泄漏到计算节点 | 确认 (2) unset 代理那行在;别在脚本后面又 source 了带 proxy 的 profile |
| 多机起不来 / 卡在 rendezvous | MASTER_ADDR/PORT 或 NNODES/NODE_RANK 没拿到 | 用平台注入的 `SENSECORE_*`/`MASTER_*`,别写死 |
| import 段错误 / 符号不匹配 | 用挂载源码去覆盖了镜像里的**编译扩展** | 编译算子保持和镜像一致;只用挂载源码覆盖纯 Python(见 §2 注意点) |
| 找不到 config / 路径错 | 没 `cd` 仓库根 | 保留 (1);config 用相对仓库根的路径 |

## 7. 完整可跑参考脚本(自包含,只改顶部 5 个变量)

> 这份是**完整的、能直接 `bash` 起来的形状**,不是片段。eSEN 读不到本项目仓库,所以这里给全文。
> 你只需要改最上面 `(A)~(E)` 五个变量,其余(代理清理 / PYTHONPATH / 自检 / torchrun rendezvous)是跨项目通用的样板,**别动**。

```bash
#!/usr/bin/env bash
###############################################################################
# eSEN Muon training -- 一行命令在"不含 Muon 的集群镜像"上提交(不重建镜像)。
# 提交命令(填进 SenseCore 任务的"启动命令"框):
#   bash '/mnt/afs/eSEN/scripts/train/submit_muon.sh'
###############################################################################
set -euo pipefail

# ===================== 只改这 5 行(eSEN 自己的值) =====================
REPO=/mnt/afs/eSEN                          # (A) eSEN 仓库根(在共享盘 /mnt/afs 下)
SRC=$REPO/src                               # (B) "含顶层包"的那层:确认 $SRC/<pkg>/__init__.py 存在
RUN_DIR=/mnt/afs/eSEN/checkpoints           # (C) checkpoint/日志输出目录(共享盘,多机可见)
CONFIG=configs/muon/esen_direct_muon.yml    # (D) optimizer 已设成 eSEN Muon 的 config(相对 $REPO)
IDENT=esen_muon_direct_run1                 # (E) 本次实验名
PKG_PROBE="import inspect, esen.optim.muon as m; assert 'HybridMuon' in inspect.getsource(m)"
                                            #     ^ 改成 eSEN 新代码的真实"模块 + 标志符号"
# ======================================================================

cd "$REPO"

# 清掉登录机泄漏进来的代理(loopback 隧道在计算节点是死的 -> 否则 wandb 数据面连不上)
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

# 核心:把挂载源码顶到镜像旧包前面(绝不 pip install)
export PYTHONPATH=$SRC:${PYTHONPATH:-}

# 自检:证明 import 到的是"挂载的新源码"(带 Muon),不是镜像里的旧包。失败立即退出。
python -c "$PKG_PROBE; print('[submit] pkg OK ->', m.__file__)"

# 分布式启动:GPU 数 / 多机 rendezvous 全部取平台注入的 SENSECORE_*/MASTER_* 变量
torchrun \
  --nproc_per_node "${SENSECORE_ACCELERATE_DEVICE_COUNT:-8}" \
  --nnodes        "${SENSECORE_PYTORCH_NNODES:-1}" \
  --node_rank     "${SENSECORE_PYTORCH_NODE_RANK:-0}" \
  --master_addr   "${MASTER_ADDR:-127.0.0.1}" \
  --master_port   "${MASTER_PORT:-29500}" \
  "$REPO/train.py" \
    --config "$CONFIG" \
    --run-dir "$RUN_DIR" \
    --identifier "$IDENT"
#   ^^^ (F) 这三行是 eSEN 训练入口 + 它的 CLI 参数。务必换成 eSEN 真实的入口/参数
#           (见 §8 第 2 步:不要凭空发明,去 eSEN 仓库里找现成的启动方式照搬)。
```

## 8. 让 eSEN 从 0 写出这个脚本的步骤(发现 → 构建 → 自检)

**关键心法:别从零发明启动命令。eSEN 几乎肯定已有一套跑训练的方式(配 AdamW 的),Muon 提交脚本 = 拿 eSEN 现成的启动命令,套上"cd + 清代理 + PYTHONPATH + import 自检"这层壳,再把 config 指向一个 Muon config。**

1. **定位 (A)(B) —— 仓库根与源码层**
   - 在 eSEN 仓库找顶层包目录:`find . -name __init__.py | head` 或看 `pyproject.toml`/`setup.py` 的 `packages`/`package_dir`。
   - `$SRC` 必须使得 `python -c "import <pkg>"` 能解析到挂载源码;验证:`PYTHONPATH=$SRC python -c "import <pkg>, inspect; print(<pkg>.__file__)"` 指向 `$SRC/...` 而不是镜像 site-packages。

2. **定位 (F) —— eSEN 现成的训练启动方式(最重要)**
   - 找现有启动脚本/文档:`ls scripts/ */launch* */train*`、`grep -rn "torchrun\|python -m\|def main\|argparse\|LightningCLI\|hydra" --include=*.py --include=*.sh`。
   - 看清它是 `torchrun <entry>.py ...` 还是 `python -m <pkg>.train ...` 还是 Hydra/Lightning。**照它的形状填进脚本的 torchrun 段**(如果 eSEN 本就用 torchrun,直接替换入口与参数;若用 `python -m`,把 torchrun 那行换成对应形式,但保留 `--nproc_per_node` 等 rendezvous 参数)。
   - 把它原来的 `--config`/`--run-dir`/实验名参数对应过来。

3. **定位 (D) —— 一个 Muon config**
   - 复制 eSEN 一个现有 AdamW 训练 config,把 optimizer 段换成 eSEN 的 Muon(参数见 `MUON_PORTING_NOTES.md` §7)。确认 `optimizer` 字段名与 eSEN 框架一致。

4. **写自检 (PKG_PROBE)**
   - 选 eSEN 新代码里一个一定存在的符号(如 `HybridMuon` 类名)。`m` 必须是定义它的模块。失败 `assert` 会立刻退出,省得白跑。

5. **自检脚本(提交前在交互式 pod / 任意能 import 的环境跑一遍)**
   ```bash
   bash -n scripts/train/submit_muon.sh                 # 语法
   PYTHONPATH=$SRC python -c "$PKG_PROBE; print(m.__file__)"   # import 到的是挂载源码?
   # 干跑 1~2 step(可临时把 epoch/步数调到极小)确认入口/参数/config 通
   ```
   - 三项都过,再去 Web UI 用 `bash '<脚本绝对路径>'` 提交、设备数填 N。

6. **常见陷阱**:见 §6。最高频两个 —— import 到旧包(查 §2/自检 `__file__`)、wandb 面板空(查 §2 的 unset 代理)。

> 配套阅读:Muon 优化器本身怎么实现/对齐,见同目录 `MUON_PORTING_NOTES.md`。

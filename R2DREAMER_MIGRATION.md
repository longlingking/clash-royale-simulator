# R2-Dreamer × Clash Royale 迁移合并文档

> 用途：在**另一台电脑（B 机）**上由 Agent 依据本文档，把本机（A 机）已完成的
> R2-Dreamer（世界模型）接入 clash-royale-simulator 的迁移完整复现，并与 B 机自己的
> 本地改动安全合并。
>
> 最后更新：A 机 2026-08-16（git HEAD `5d6a710`，与 `origin/main` 一致）。

---

## 1. 背景与两机现状

- 仓库：`https://github.com/longlingking/clash-royale-simulator.git`，分支 `main`。
- **A 机（本机）**：`HEAD == origin/main == 5d6a710`，所有历史提交均已推送；
  r2dreamer 接入工作**全部未提交**（见第 2 节清单）。
- **B 机**：clone 自同一仓库，有本地更新，但**无法推送到远程**（原因未知，可能是
  大文件超限 / 分支保护 / 凭证 / 纯离线，见第 7 节排查）。
- 目标：
  1. 把 A 机的 r2dreamer 接入以代码形式同步到远程仓库；
  2. B 机拉取后与自己的改动合并、复现迁移；
  3. 全程不产生难以修复的历史问题（尤其是**大文件**）。

### 合并策略结论（先回答核心问题）

**推荐：先在 A 机把"纯代码"提交并推送，再到 B 机 pull --rebase 解决冲突。** 理由：

1. A 机的改动是**纯附加式**（1 个新目录 + 4 个新脚本 + 1 个图片 + `.gitignore` 追加 3 行），
   与 B 机改动的冲突面极小（唯一可能重叠的是 `.gitignore`，且双方都是追加行，
   三路合并几乎总能自动完成）。
2. A 机的 push **确定性可行**：远程本来就是从 A 机推的，只要不把 844MB 的
   `r2dreamer/logdir/` 提交进去（其中 checkpoint 每个 121MB，超过 GitHub 单文件
   100MB 硬限制，**一旦提交 push 必失败**），push 不会遇到任何障碍。
3. B 机在自己本地解决冲突上下文最全：自己的提交 + 我们的新代码都在工作区里，
   验证（smoke 测试）也直接可做。
4. 反方向（先把 B 机改动搬到 A 机合并）需要传输 B 机的提交（bundle/patch），
   步骤更多、出错面更大；且如果 B 机 push 失败是因为大文件，问题迟早还要回
   B 机解决，绕不开。

**唯一前提**：B 机必须能从远程 fetch/pull。若 B 机完全离线，改用第 8 节的
`git bundle` 兜底方案。

---

## 2. A 机未提交变更清单（迁移的全部内容）

| 路径 | 类型 | 大小 | 说明 |
|---|---|---|---|
| `r2dreamer/` | 新目录 | 692 KB（代码） | R2-Dreamer（ICLR 2026 官方实现）整体 vendor 进仓库 |
| `r2dreamer/envs/cr.py` | **新增** | 142 行 | Clash Royale 环境适配器（核心文件，见第 3 节） |
| `r2dreamer/configs/env/cr.yaml` | **新增** | — | CR 环境 Hydra 配置 |
| `r2dreamer/runs/cr.sh` | **新增** | — | 训练入口脚本 |
| `r2dreamer/envs/__init__.py` | **修改** | — | `make_env()` 增加 `suite == "cr"` 分支（上游唯一被改的代码文件） |
| `r2dreamer/` 其余文件 | 上游原样 | — | 上游官方仓库文件，无改动 |
| `smoke_cr.py` | 新增 | 20 行 | 环境链路冒烟测试（**B 机验收用**） |
| `bench_sim.py` | 新增 | 69 行 | 模拟器单机 vs 8 并行吞吐基准 |
| `bench_battle.py` | 新增 | 32 行 | 战场拥挤度 vs step 耗时基准 |
| `draw_r2dreamer_arch.py` | 新增 | 161 行 | 绘制架构图（依赖 matplotlib） |
| `r2dreamer_architecture.png` | 新增 | 328 KB | 架构图产物（可选提交） |
| `.gitignore` | **修改** | +3 行 | 追加 `r2dreamer/logdir/`、`r2dreamer/.hydra/`、`.pylibs/` |

**明确不提交**（必须保持被 ignore）：

| 路径 | 大小 | 原因 |
|---|---|---|
| `r2dreamer/logdir/` | 844 MB（7 个 121MB 的 `latest.pt` + tfevents） | 单文件超 GitHub 100MB 限制；训练产物不进版本库 |
| `.pylibs/` | ~几十 MB | A 机 conda 环境只读的本地变通（vendored 依赖），不可移植 |
| `r2dreamer/r2dreamer.egg-info/`、`__pycache__/` | — | 嵌套 `.gitignore` 已忽略（`*.egg-info/`），顶层 `__pycache__/` 全局忽略 |

> 提交前自查命令（A 机/B 机都适用）：
> ```bash
> git add -n .            # 逐行检查，确认没有任何 logdir/.pylibs/egg-info 条目
> git ls-files -z | xargs -0 -r ls -l | sort -k5 -n | tail -5   # 确认最大文件 < 100MB
> ```

---

## 3. 集成机制（B 机 Agent 理解与验证用）

### 3.1 观测空间（observation dict，numpy）

| key | shape | dtype | 用途 |
|---|---|---|---|
| `grid` | `(32, 18, 15)` | float32 | CNN key（`cnn_keys: 'grid'`）；15 通道已按 `_GRID_SCALE` 归一化到约 [0,1]，`nan_to_num` 兜底 |
| `hand` | `(5,)` | float32 | MLP key，手牌卡 id 0..12 |
| `elixir` | `(1,)` | float32 | MLP key，0..10 |
| `is_first` / `is_last` / `is_terminal` | `()` | bool | trainer/agent 必需标志位 |

### 3.2 动作空间

- `gym.spaces.Box(low=0, high=1, shape=(5, 32, 18))`，标记 `multi_discrete=True`。
- 实际是**扁平的多 one-hot 向量（55 维）**：`[slot(5) | y(32) | x(18)]`；
  `step()` 内按三段 argmax 解码为 `(slot, y, x)` 传给 `CREnv.step`。
- r2dreamer 侧**不需要** `OneHotAction` 包装（`envs/__init__.py` 的 `cr` 分支已注明）。

### 3.3 环境构建链路

```
envs/__init__.py make_env()            # suite = config.task.split('_')[0] = "cr"
  └─ envs.cr.ClashRoyale(opponent=config.opponent, seed=config.seed+id, speed=config.speed)
       └─ 内部：os.environ SDL_VIDEODRIVER=dummy / SDL_AUDIODRIVER=dummy（无头 pygame）
       └─ sys.path 注入 <repo>/.pylibs 与 <repo>/src/clasher_new，并 chdir 到 src/clasher_new
       └─ CREnv(opponent_model=..., speed=...)   # gymnasium 版模拟器
  └─ wrappers.TimeLimit(env, config.time_limit // config.action_repeat)
  └─ wrappers.Dtype(env)
```

- `cr.py` 通过 `_HERE` 反推仓库根目录（`<repo>/r2dreamer/envs/` → `<repo>`），
  **布局不变则换机器无需改路径**；游戏目录可用环境变量 `CLASHER_SIM_DIR` 覆盖。
- `chdir` 副作用：`import envs.cr` 之后进程 CWD 变成 `src/clasher_new`
  （card JSON 相对 CWD 读取）。顶层脚本注意：**从仓库根运行、import 后再做自己的路径操作要小心**。
- opponent 取值：`random`（默认）| `script:bridge_rush` | `script:bridge_rush_left`
  | `script:defender` | stable-baselines3 PPO checkpoint 文件路径（懒加载）。

### 3.4 关键配置（configs/env/cr.yaml）

- `task: cr_clashroyale`（→ suite `cr`）、`steps: 5e5`、`env_num: 1`、`eval_episode_num: 2`
- `action_repeat: 1`、`time_limit: 600`：每 env step = 30 游戏帧 = 0.5s 战斗时间；
  模拟器 180s 强制平局、300s 结束 → 一局 600 步（TimeLimit 兜底）
- `encoder/decoder`: `mlp_keys: '^(hand|elixir)$'`、`cnn_keys: 'grid'`

### 3.5 训练入口（runs/cr.sh）

```bash
python train.py env=cr env.steps=$STEPS logdir=logdir/${DATE}_${METHOD}_cr \
    model.rep_loss=r2dreamer model.compile=False device=cpu \
    batch_size=16 batch_length=64 trainer.train_ratio=64 seed=0
```

- `$STEPS` 默认 500000，可传参：`bash runs/cr.sh 2000`（B 机短验用）。
- 无 GPU 机器默认 `device=cpu, compile=False`；有 GPU 时可去掉这两个 override。
- checkpoint 存到 `logdir/.../latest.pt`（121MB），另有 `save_every: 1e4`。
- 断点续训：顶层 `configs.yaml` 有 `resume: ''` 字段，可
  `resume=/abs/path/to/logdir/xxx/latest.pt`（**注意 Hydra 会 chdir 到 run dir，
  路径用绝对路径**）。

---

## 4. 依赖与环境（A 机已验证事实 + B 机安装）

### 4.1 A 机实际环境（供对照）

- conda env：`r2dreamer`，Python 3.11（`/home/longling/miniconda3/envs/r2dreamer`）
- `r2dreamer` 以 **editable** 方式安装（`pip show r2dreamer` → Editable project location 指向仓库内 `r2dreamer/`）
- 已装（pip list 实测）：`torch 2.8.0`、`torchrl 0.9.2`、`tensordict 0.9.1`、
  `gymnasium 1.2.0`、`numpy 1.26.0`、`einops 0.3.0`、`moviepy 1.0.3`、
  `ruamel.yaml 0.17.4`、`setuptools 77.0.3`、`tensorboard 2.21.0`、
  `pygame 2.6.1`、`fastcore 2.2.12`；`stable_baselines3` 可导入（不在 pip list 常规条目内，属 conda/手动安装，B 机直接 pip 装即可）。
- `matplotlib` **不在** env 里 → A 机的 `draw_r2dreamer_arch.py` 借道 `.pylibs` 的 matplotlib；B 机直接 `pip install matplotlib` 即可，不用复刻 `.pylibs`。

### 4.2 B 机安装命令（推荐，已验证等价）

```bash
conda create -n r2dreamer python=3.11 -y
conda activate r2dreamer
cd <repo>/r2dreamer
pip install -e .                        # 按 pyproject 装齐核心依赖（torch 2.8.0 等）
pip install pygame fastcore stable-baselines3 matplotlib
cd <repo>                               # 回到仓库根
```

- Python 版本硬约束：`requires-python >=3.11,<3.12`，**不要用系统 python3**。
- numpy 版本：sim README 写 `numpy==1.26.4`，pyproject 钉 `1.26.0`，同为 1.26 系，无冲突。
- 若 B 机有 GPU：不需要 `.pylibs`，也不需要 `--user` 之类的历史命令（仓库 README 里的
  `pip install ... --user` 是旧写法，建议用上面 conda env 方式）。

---

## 5. B 机迁移复现步骤（按序执行）

> 前置：A 机已按第 6 节完成"提交 + 推送"，B 机已 `git fetch` 到新提交。

```bash
# 0) 先处理 B 机自己的改动（第 6.2 节）：commit 或 stash，确保工作区干净
git status

# 1) 拉取 A 机的迁移
git fetch origin
git rebase origin/main          # 推荐 rebase；想保留合并记录也可 git merge origin/main
#    —— 若有冲突，见第 6.3 节 ——

# 2) 装环境（第 4.2 节）

# 3) 验收 1：冒烟测试（预期输出见第 9 节）
python smoke_cr.py              # 期望末尾打印 WRAPPER_CHAIN_OK

# 4) 验收 2：吞吐基准（可选）
python bench_sim.py             # 或 python bench_sim.py single / parallel

# 5) 验收 3：短训练（可选，CPU 上几千步几分钟内可跑完）
bash runs/cr.sh 2000
tensorboard --logdir r2dreamer/logdir   # 或看 logdir/<date>_r2dreamer_cr/metrics.jsonl

# 6) 把 B 机自己的改动推回远程（此时远程已含 A 机迁移）
git push origin main
```

---

## 6. 合并执行流程（完整版）

### 6.1 A 机：提交并推送（建议拆两个提交）

```bash
cd <repo>   # A 机仓库根
git add -n .                              # ★先自查：确认无 logdir/.pylibs/egg-info
git add .gitignore r2dreamer/
git commit -m "Vendor R2-Dreamer with Clash Royale env adapter (envs/cr.py, configs/env/cr.yaml, runs/cr.sh)"
git add bench_battle.py bench_sim.py smoke_cr.py draw_r2dreamer_arch.py \
        r2dreamer_architecture.png R2DREAMER_MIGRATION.md
git commit -m "Add r2dreamer integration scripts and migration doc"
git push origin main
```

### 6.2 B 机：先固化自己的改动

```bash
git status                          # 看自己有哪些未提交/未推送内容
# 情况 A：有未提交文件且想保留 → 提交（记得用各自的 .gitignore 排除大文件）
git add <自己的文件>
git commit -m "WIP: B machine local updates"
# 情况 B：只是临时试验 → git stash
```

### 6.3 B 机：rebase 与冲突处理

```bash
git fetch origin
git rebase origin/main
# 冲突时：
git status                          # 列出冲突文件
#   预期只有 .gitignore 可能冲突（双方都是追加行，合并结果 = 保留双方规则）
#   若 B 机也新建了同名文件（bench_*.py / r2dreamer/ 内文件），逐个取舍
git mergetool                       # 或手动编辑后 git add <file>
git rebase --continue
# 完成后强制验证：
python smoke_cr.py
git push origin main
```

### 6.4 如果 B 机 push 还是失败（排查表）

| 现象 | 原因 | 处理 |
|---|---|---|
| `remote: error: File ... is 121.00 MB; this exceeds GitHub's file size limit of 100.00 MB` | 大文件进了提交 | 若只是工作区：加 `.gitignore` 后 `git rm --cached`；若已进历史：`git filter-repo` / BFG 清理（见第 8 节） |
| `protected branch` / `403` | 分支保护或凭证 | 推 feature 分支 + PR，或换带 push 权限的 token |
| `failed to push some refs`（非快进） | 远程有新提交 | 先 `git pull --rebase origin main` 再推 |
| 完全无网络 | 离线 | 用第 8 节 `git bundle` |

---

## 7. 兜底方案（B 机离线 / 无法访问远程）

方向 A → B（把 A 机的迁移带给 B 机）——在 A 机：

```bash
git bundle create r2dreamer-migration.bundle origin/main..HEAD
# 把该 bundle 文件拷到 B 机（U 盘/网盘）
```

在 B 机：

```bash
git fetch /path/to/r2dreamer-migration.bundle main:refs/remotes/origin/main
git rebase origin/main
```

方向 B → A（把 B 机的改动带回 A 机合并）——在 B 机打包自己的提交：

```bash
git bundle create my-updates.bundle origin/main..HEAD
# 拷到 A 机后：
git fetch /path/to/my-updates.bundle main:refs/heads/b-machine
git merge b-machine                 # 在 A 机解决冲突、验证、push
```

---

## 8. 坑位清单（复现时最容易翻车的地方）

1. **121MB checkpoint / 844MB logdir**：超过 GitHub 100MB 单文件硬限制。
   `git add -A` / `git add .` 前先 `git add -n .` 检查。logdir 已由
   `r2dreamer/.gitignore`（`logdir/`）+ 顶层 `.gitignore` 双保险忽略。
2. **顶层三个脚本硬编码了 A 机路径**：`smoke_cr.py`、`bench_sim.py`、
   `bench_battle.py` 第一行 `sys.path.insert(0, '/home/longling/随便玩玩/...')`。
   B 机若已 `pip install -e ./r2dreamer`，**可删掉这行**；不想改脚本就改成
   B 机的仓库绝对路径。路径含中文（`随便玩玩`）在 Python 3 可用但脆弱。
3. **`import envs.cr` 会 chdir 到 `src/clasher_new`**：import 之后 CWD 变了，
   后续相对路径操作要当心；脚本统一从仓库根运行。
4. **`.pylibs` 不可移植**：它是 A 机 conda 只读环境的变通产物，B 机用正常
   `pip install` 替代，**不要提交、不要复制**。
5. **`draw_r2dreamer_arch.py` 依赖 matplotlib**：A 机靠 `.pylibs`，B 机
   `pip install matplotlib`；若机器上没有 `.pylibs` 目录且不想装 matplotlib，
   跳过该脚本即可（png 是产物）。
6. **Python 3.11 限定**：`requires-python >=3.11,<3.12`；torch 2.8.0 钉死，
   不要随意升级。
7. **CPU 训练慢**：cr.sh 默认 `device=cpu, compile=False`；验证用
   `bash runs/cr.sh 2000` 级别的小步数，不要一上来跑 5e5。
8. **stable-baselines3 是 sim 的硬依赖**：`src/clasher_new/environment.py` 顶层
   `from stable_baselines3.common.env_checker import check_env`，**即使只用
   random opponent 也必须装 sb3**（脚本对手/checkpoint 对手是懒加载，random 不是）。
9. **Hydra chdir 到 run dir**：`resume=` 用绝对路径；`logdir` 相对路径只对
   启动时的 CWD 有效。
10. **`.DS_Store` 已被仓库跟踪**（历史遗留），与本迁移无关，别顺手删了造成噪音。
11. **`r2dreamer/runs/*.sh`、configs 里其它环境的文件是上游原样**，全部随
    vendor 提交，属预期，不要"清理"掉。
12. numpy 版本：1.26 系即可（sim 与 pyproject 分别钉 1.26.4 / 1.26.0，同系兼容）。

---

## 9. 验证清单与预期输出

### 9.1 smoke_cr.py（A 机实测输出，B 机应一致）

```
reset keys: ['elixir', 'grid', 'hand', 'is_first', 'is_last', 'is_terminal']
  grid (32, 18, 15) float32
  hand (5,) float32
  elixir (1,) float32
  is_first () bool
  is_last () bool
  is_terminal () bool
step 0: r=0.000 done=False
step 1: r=0.000 done=False
step 2: r=0.000 done=False
step 3: r=0.000 done=False
step 4: r=-0.247 done=False
WRAPPER_CHAIN_OK
```

判定：末尾 `WRAPPER_CHAIN_OK`，reset 的 key 集合一致即通过。

### 9.2 短训练（runs/cr.sh 2000）

- 命令：`bash runs/cr.sh 2000`（无 GPU 默认 cpu，几分钟量级）
- 判定：`logdir/<MMDD>_r2dreamer_cr/` 下出现 `metrics.jsonl`、`latest.pt`、
  `console.log`；console 出现
  `Encoder CNN shapes: {'grid': (32, 18, 15)}`、
  `Encoder MLP shapes: {'elixir': (1,), 'hand': (5,)}` 即环境解析正确；
  训练循环能打印 `episode/eval_score` 与 `fps` 即通过。
- 可选：`tensorboard --logdir r2dreamer/logdir` 看曲线。

### 9.3 回归（B 机自己的功能）

- 确认 B 机原有的训练/评估/自对弈流程在 rebase 后照常运行（A 机未改动
  `src/` 任何文件，理论上零影响；若 B 机曾改过 `src/`，以 B 机版本为准）。

---

## 10. B 机 Agent 执行检查表（照此顺序交付）

- [ ] 1. `git status` 确认 B 机本地改动，先 commit/stash 固化
- [ ] 2. `git fetch origin && git rebase origin/main`，冲突按 6.3 处理
- [ ] 3. 创建 conda env `r2dreamer`（python=3.11），`pip install -e ./r2dreamer`
      + `pip install pygame fastcore stable-baselines3 matplotlib`
- [ ] 4. `python smoke_cr.py` → `WRAPPER_CHAIN_OK`（若脚本里还是 A 机硬编码路径，
      先删掉 sys.path 行或改成 B 机路径）
- [ ] 5. （可选）`bash runs/cr.sh 2000` 短训通过
- [ ] 6. 确认 `git add -n .` 无大文件后 `git push origin main`
- [ ] 7. 汇报：合并方式（rebase/merge）、冲突文件清单、验证结果

> 若 B 机无法 fetch/pull：改用第 7 节 bundle 流程；若 push 被拒：按 6.4 排查表处理。

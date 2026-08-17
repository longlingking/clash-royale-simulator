# Future Work

> 待实现清单。已讨论、方向明确但尚未落地的改进。

## 1. IsaacLab 式按对手比例采样（per-env 混合对手）

> ✅ 已实现（`src/clasher_new/self_play.py`：`OpponentEpisodeWrapper` + `OpponentPool.child` + `make_opponent_vec_env`；`train.py` 已切换到 `SubprocVecEnv(K=8)` 并按 `n_steps=2048//K` 保持 batch=2048）。测试：`tests/test_opponent_sampling.py`。

**目标**：把 self-play 从"窗口级混合"改成"批次内混合"，降低 PPO 梯度方差、抑制 self-play 的遗忘/震荡。

**现状问题**：
- `OpponentSwapCallback`（`src/clasher_new/self_play.py`）`swap_every=2048`，而 SB3 单 env 的 `n_steps` 默认正好也是 2048。
- 结果是**每一次 PPO 更新喂的 2048 步全部来自同一个对手**，下次更新又整体换成另一个对手。策略和 value function 在 update 之间来回追对手 → rock-paper-scissors / forgetting 在批次粒度上的表现。

**改法（IsaacLab 落点：N 个 env 各自随机抽配置，rollout 后拼出的 batch 天然按比例混合）**：
1. `SubprocVecEnv(n_envs=K)`，K ≥ 对手类型数。
2. 每个子 env 包 wrapper，在 `reset()` 时从 `OpponentPool.pick()` 重新抽对手（per-episode 采样，等价于每次抽地形）。
3. 去掉对单个 `self.env.opponent` 赋值的 `OpponentSwapCallback`。

```python
class OpponentEpisodeWrapper(gym.Wrapper):
    def __init__(self, env, pool):
        super().__init__(env); self.pool = pool
    def reset(self, **kw):
        self.unwrapped.opponent = self.pool.pick()
        return super().reset(**kw)
# make_vec_env(lambda: OpponentEpisodeWrapper(CREnv(), pool), n_envs=K, ...)
```

每个 PPO batch = K 个不同对手的轨迹混合，分布正好等于权重（0.6 recent / 0.2 base / 0.2 fixed）。
注意 `make_vec_env` 默认 `DummyVecEnv`（不是 SubprocVecEnv，需显式指定），且 `SubprocVecEnv` 默认用 forkserver 启动；每个子进程自带 pool/模型缓存，deck 是各自副本，不会互相踩。

## 2. 自适应对手权重（真正的数据效率来源）

> ✅ 已实现（`src/clasher_new/self_play.py`：`OpponentPool._weighted_candidates` / `set_priorities`、`adapt_priorities` 纯函数、`AdaptiveWeightCallback`；`OpponentEpisodeWrapper` 通过 `info['opponent']/['won']` 上报每局结果；`train.py` 已挂 `AdaptiveWeightCallback`，默认 `--adap-update-every 50k / --adap-alpha 0.3 / --adap-floor 0.1`）。测试：`tests/test_adaptive_weights.py`（10 个，含真实 `model.learn` 冒烟）。

**注意点**：固定比例会稀释信号。fixed 脚本是"容易的靶子"，agent 已能高胜率打过时，混太多脚本样本是拿宝贵样本喂给没学习量的目标——这叫浪费。IsaacLab 用 curriculum 而非静态均匀混合，同理。

**改法**：把 `weights` 从固定值改成基于 winrate 的自适应（Prioritized Fictitious Self-Play / AlphaStar 思路）：
- 被当前 agent 打得越狠的旧快照 → 采样权重越低；
- 越是能赢当前 agent 的对手 → 权重越高；
- 自动把训练资源从"太简单"挪到"刚好有学习价值"的对手上。

**落地形式**：信号直接来自训练对局本身（wrapper 每局上报 `(opponent, won)`，主进程回调累计，每 `update_every` 步把 winrate 转成 per-candidate priority，`p = max(floor, 1−winrate)` 再指数平滑，经 `env_method('set_priorities', ...)` 广播给 8 个子进程）。`pick()` 权重 = `桶基础权重 × priority`：桶内按 priority 加权、整个桶被碾压时按平均 priority 缩水。全 priority=1.0 时与旧的均匀两级抽样逐位一致。

**不变**：`BestWeightCallback` 用的固定评审团（`make_eval_crowd`）保持固定——它仍是选 `best_model` 的唯一稳定标尺。

## 优先级小结

| 项目 | 收益 | 优先级 |
|------|------|--------|
| per-env 比例混合（#1） | 抹平震荡，训练更稳 | 第一步 |
| 自适应权重（#2） | 纯数据效率，省样本 | 第二步 |

## 相关

- `src/clasher_new/train.py` — PPO 训练入口
- `src/clasher_new/self_play.py` — `OpponentPool` / `OpponentSwapCallback` / `BestWeightCallback`
- README.md "To-do list" — 已有的模型架构 / 评价基准 / 环境改进想法

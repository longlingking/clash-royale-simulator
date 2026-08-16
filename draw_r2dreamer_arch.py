"""Render the architecture diagram of the R2-Dreamer x Clash Royale integration.

Output: r2dreamer_architecture.png (next to this script).
Layout is hand-tuned on a 0..100 x 0..104 canvas; keep boxes on the grid.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_HERE, ".pylibs", "mplconfig"))
sys.path.insert(0, os.path.join(_HERE, ".pylibs"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BLUE_BG, BLUE_ED = "#dbe9f7", "#1f77b4"
GRN_BG, GRN_ED = "#e2f0d9", "#2e7d32"
ORG_BG, ORG_ED = "#fde8d7", "#e07b00"
GRAY = "#666666"

fig, ax = plt.subplots(figsize=(17, 11.5), dpi=150)
ax.set_xlim(0, 100)
ax.set_ylim(0, 104)
ax.axis("off")


def box(x, y, w, h, title, lines, bg, ed, title_size=9.5, body_size=8.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=bg, ec=ed, lw=1.6, zorder=3))
    ax.text(x + w / 2, y + h - 3.2, title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", color=ed, zorder=4)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 6.6 - i * 3.4, ln, ha="center", va="center",
                fontsize=body_size, color="#222222", zorder=4)


def arrow(pts, color=GRAY, lw=1.6, ls="-", label=None, lab=(0, 0), lab_color=None):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=2, solid_capstyle="round")
    ax.annotate("", xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls), zorder=2)
    if label:
        ax.text(lab[0], lab[1], label, fontsize=8.2, color=lab_color or color,
                ha="center", va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.9))


# ============ left column: environment side (blue) ============
box(2, 88, 28, 8, "Opponent (per worker process)",
    ["random | script: bridge_rush / defender | SB3 checkpoint"],
    BLUE_BG, BLUE_ED, title_size=8.6, body_size=7.4)
box(2, 58, 28, 26, "CREnv  Simulator  (battle.py)",
    ["Troops / buildings / spells / towers",
     "decision every 30 frames = 0.5 s battle time",
     "reward: tower HP + crowns (5) + win/lose (10)",
     "natural end: draw at 180 s, finish at 300 s"],
    BLUE_BG, BLUE_ED)
box(2, 30, 28, 22, "ClashRoyale adapter  (envs/cr.py)",
    ["grid channel normalize + nan/inf cleanup",
     "obs: grid(32,18,15) hand(5) elixir(1)",
     "action: flat one-hot (5|32|18) -> (slot, y, x)",
     "flags: is_first / is_last / is_terminal",
     "legacy gym API (4-tuple) for r2dreamer"],
    BLUE_BG, BLUE_ED)

# ============ middle column: agent (green) ============
box(34, 88, 32, 8, "MultiEncoder",
    ["ConvEncoder(grid 32x18x15)  +  MLP(hand 5, elixir 1)",
     "-> embed (B, T, 384)"],
    GRN_BG, GRN_ED, title_size=9.0, body_size=8.0)
box(34, 62, 32, 22, "RSSM  World Model",
    ["Deter: block-GRU transition (stoch, deter, prev_action)",
     "posterior: obs_net(deter + embed)",
     "prior: img_net(deter)   |   kl: dyn + rep (free 1.0)",
     "stoch (B, T, 32, 16) + deter (B, T, 2048)"],
    GRN_BG, GRN_ED)
box(34, 40, 32, 16, "Prediction Heads  +  R2-Dreamer",
    ["reward head (symexp_twohot)",
     "cont head (binary)",
     "Projector(feat -> embed)  +  Barlow Twins loss",
     "(decoder-free representation learning)"],
    GRN_BG, GRN_ED, title_size=9.0, body_size=7.8)
box(34, 16, 32, 18, "Actor - Critic",
    ["actor: MultiOneHotDist (5|32|18, 55 dims, unimix)",
     "critic + slow critic: symexp_twohot (EMA 0.02)",
     "imagined rollout: horizon 15, lambda-return"],
    GRN_BG, GRN_ED)

# ============ right column: training pipeline (orange) ============
box(70, 88, 28, 8, "ReplayBuffer  (torchrl)",
    ["LazyTensorStorage 5e5 · SliceSampler",
     "batch B x (L+1), latent (stoch/deter) written back"],
    ORG_BG, ORG_ED, title_size=9.0, body_size=7.8)
box(70, 62, 28, 22, "OnlineTrainer",
    ["loop: agent.act -> env step -> buffer -> update",
     "train_ratio gates update frequency",
     "eval_every · tensorboard / metrics.jsonl",
     "checkpoint: latest.pt (agent + optims)"],
    ORG_BG, ORG_ED)
box(70, 40, 28, 16, "Losses",
    ["world model: dyn KL + rep KL + rew + con",
     "representation: barlow (invariance + redundancy)",
     "imagination: policy + value + repval"],
    ORG_BG, ORG_ED, title_size=9.0, body_size=7.8)
box(70, 16, 28, 18, "Evaluation",
    ["eval envs (n_eval) with greedy (mode) actions",
     "metrics: episode/eval_score · eval_length",
     "video / open-loop prediction (optional)"],
    ORG_BG, ORG_ED)

# ============ internal arrows (env column) ============
arrow([(16, 88), (16, 84.6)], lw=1.2)
arrow([(16, 58), (16, 52.6)], lw=1.2)

# ============ data-flow arrows ============
# obs: adapter -> encoder
arrow([(30, 41), (34, 91.5)], label="obs: grid(32,18,15) hand(5) elixir(1) + flags",
      lab=(32.5, 66), lab_color=BLUE_ED, lw=1.8)
# embed
arrow([(50, 88), (50, 84)], label="embed (B,T,384)", lab=(58.5, 86))
# feat
arrow([(50, 62), (50, 56)], label="feat (B,T,2560)", lab=(58.5, 59))
# heads -> actor-critic
arrow([(50, 40), (50, 34)], label="imag rollout", lab=(58, 37))
# action loop: actor -> adapter
arrow([(34, 18), (30, 31.5)], label="action (B,55) multi one-hot -> (slot,y,x)",
      lab=(18, 13.5), lab_color=BLUE_ED, lw=1.8)
# transitions: adapter -> buffer (top corridor, y=98.6)
arrow([(16, 52), (16, 98.6), (84, 98.6), (84, 96.6)],
      label="transition: obs + action + reward + flags",
      lab=(50, 97.5), lab_color=ORG_ED)
# latent write-back: rssm -> buffer
arrow([(66, 70), (70, 88)], ls="--", label="stoch/deter write-back",
      lab=(68.5, 79), lab_color=ORG_ED)
# buffer -> trainer
arrow([(84, 88), (84, 84)], label="batch B x (L+1)", lab=(92.5, 86))
# trainer -> losses
arrow([(84, 62), (84, 56)])
# gradients: trainer -> agent
arrow([(70, 66), (66, 66)], lw=2.4, color="#c0392b",
      label="gradients (LaProp + AGC)", lab=(66.5, 64.5), lab_color="#c0392b")
# eval: agent -> eval
arrow([(66, 20), (70, 21.5)], ls="--", label="eval policy (frozen)",
      lab=(69.5, 13.5), lab_color=ORG_ED)
# eval -> env (bottom corridor, y=7)
arrow([(84, 16), (84, 7), (16, 7), (16, 30)], ls="--", label="eval actions (greedy mode)",
      lab=(50, 4.5), lab_color=ORG_ED)

# ============ titles (above the canvas top edge, ylim=104) ============
ax.text(50, 103.2, "R2-Dreamer (DreamerV3 world model)  x  Clash Royale Simulator",
        ha="center", va="center", fontsize=15, fontweight="bold", color="#111111")
ax.text(50, 101.0, "train.py  (hydra: env=cr, model=size12M)  |  10,208,694 params  |  cpu / cuda",
        ha="center", va="center", fontsize=9, color="#555555")

plt.savefig(os.path.join(_HERE, "r2dreamer_architecture.png"),
            bbox_inches="tight", facecolor="white")
print("saved:", os.path.join(_HERE, "r2dreamer_architecture.png"))

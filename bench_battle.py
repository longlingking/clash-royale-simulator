"""Measure battle step cost vs battlefield complexity (crowded games)."""
import sys, time
sys.path.insert(0, '/home/longling/随便玩玩/clash-royale-simulator/r2dreamer')
import numpy as np
from envs.cr import ClashRoyale

env = ClashRoyale()
env.reset()

def crowded_action():
    # always deploy a card somewhere (both players crowd the board)
    a = np.zeros((5, 32, 18), dtype=np.float32)
    a[0][np.random.randint(5)] = 1.0
    a[1][np.random.randint(32)] = 1.0
    a[2][np.random.randint(18)] = 1.0
    return a

times, counts = [], []
for i in range(300):
    t0 = time.time()
    o, r, d, _ = env.step(crowded_action())
    times.append(time.time() - t0)
    counts.append(int((o['grid'] > 0).any(axis=-1).sum()))
    if d:
        env.reset()
times = np.array(times); counts = np.array(counts)
print(f"[crowded] 300 steps: avg {times.mean()*1000:.1f} ms/step | occupied cells avg {counts.mean():.0f} (max {counts.max()})")
# bucket by occupied cells
for lo in range(0, 45, 10):
    m = (counts >= lo) & (counts < lo + 10)
    if m.sum() > 5:
        print(f"  cells {lo}-{lo+9}: {m.sum()} steps, avg {times[m].mean()*1000:.1f} ms/step")

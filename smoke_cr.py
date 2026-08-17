import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'r2dreamer'))
import numpy as np
from envs.wrappers import TimeLimit, Dtype
from envs.cr import ClashRoyale

env = Dtype(TimeLimit(ClashRoyale(opponent='random'), 600))
o = env.reset()
print('reset keys:', sorted(o.keys()))
for k, v in o.items():
    print(' ', k, getattr(v, 'shape', None), getattr(v, 'dtype', None))
for i in range(5):
    a = np.zeros((5, 32, 18), dtype=np.float32)
    for seg in (a[0], a[1], a[2]):
        seg[np.random.randint(len(seg))] = 1.0
    o, r, d, info = env.step(a)
    print(f'step {i}: r={r:.3f} done={d}')
    if d:
        break
print('WRAPPER_CHAIN_OK')

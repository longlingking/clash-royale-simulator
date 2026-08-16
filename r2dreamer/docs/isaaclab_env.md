# IsaacLab Environment Integration

This document explains the non-obvious parts of [`envs/isaaclab.py`](../envs/isaaclab.py):
why the reset is deferred by one step, how the `is_first` / `is_last` / `is_terminal`
flags line up, and which IsaacLab internals are being patched.

## Why defer the reset?

Dreamer-style world model training needs the **terminal observation** — the obs
emitted on the step that ends an episode — as the prediction target. IsaacLab's
default `step()` auto-resets terminated envs in the same call, overwriting the
terminal obs with the *initial* obs of the next episode. The trainer would then
see initial-obs as if it were the terminal obs and learn a wrong transition.

The patch in `_patch_env` splits this into two steps:

1. **Terminal step** — episode ends; intercept `_reset_idx` so the terminal obs is preserved.
2. **Reset step** — the trainer signals `done=True` on the next call; reset
   explicitly and overwrite obs with the initial obs of the new episode.

## Timeline

`T` denotes the **terminal step** (the step on which the episode ends inside the env).
`done` is the flag the trainer passes into `step(action, done=...)`, equal to
the previous step's `episode_done`.

```
step index            t = T-1          t = T   (terminal)         t = T+1 (reset)
------------------    ---------------  -----------------------    -----------------------
done arg (trainer)    F                F                          T   (action zeroed in env)

inside env            normal physics   episode ends;              super().step() runs a
                                       _reset_idx intercepted     junk step, then explicit
                                                                  _reset_idx + obs refill

returned obs          normal           terminal_obs               initial_obs
returned reward       r                terminal_r                 0 (zeroed)
returned term/trunc   F / F            T / F  (or F / T)          F / F  (masked by ~done)

is_first              F                F                          T
is_terminal           F                T                          F
is_last               F                T                          F
```

Key consequence: the trainer's per-step `done` is **delayed by one step** relative
to the env's internal termination. The terminal transition is captured intact at
`t=T`; the following step (`t=T+1`) begins a fresh episode with `is_first=True`.

## Flag semantics (`IsaacLabVecEnv.step`)

| Flag           | Set when                                | Source                       |
| -------------- | --------------------------------------- | ---------------------------- |
| `is_first`     | First obs of a new episode              | `done` arg from the trainer  |
| `is_terminal`  | True termination (not time-out)         | `terminated & ~done`         |
| `is_last`      | Any episode end (termination or trunc.) | `(term \| trunc) & ~done`    |
| `reward`       | Zeroed on reset steps                   | `where(done, 0, reward)`     |

The `& ~done` mask matters because the junk step that runs inside `super().step()`
during a reset can leave stale `terminated` / `truncated` bits set; those bits
were already emitted truthfully on step `T`, so we suppress them on `T+1`
to avoid double-counting an episode end.

## IsaacLab internals being patched

All of the following are tied to **IsaacLab 2.3.2** (pinned in `pyproject.toml`)
and should be revalidated whenever the version is bumped:

- **`_reset_idx` interception** — gated by `_block_reset`, active only during the
  parent `step()` call. Lets the parent step run end-of-episode bookkeeping
  without actually wiping state.
- **Reset-flag rollback** — on the reset step, `reset_terminated`,
  `reset_time_outs`, and `reset_buf` are zeroed. The returned tuple
  `(obs, reward, terminated, truncated, extras)` references the same
  `reset_terminated` / `reset_time_outs` tensors, so without this the trainer
  would see stale `True` bits from the terminal step. Note that this is not
  about preventing future auto-resets: `_get_dones()` overwrites these flags
  at the start of every step regardless.
- **Observation history rollback** — `super().step()` appends to each
  `ObservationManager` `CircularBuffer` via `compute(update_history=True)`.
  On reset steps we decrement `_pointer` and `_num_pushes` on the private
  buffer so the subsequent explicit `compute()` refills the slot via
  `is_first_push` semantics (matching default IsaacLab reset behavior). This
  touches private fields and is the most fragile part of the patch.
- **RTX re-render** — when `num_rerenders_on_reset > 0`, sensors are re-rendered
  after the explicit reset so the refreshed obs reflects the new episode's
  initial scene rather than the terminal frame.

## Camera observation injection

For tasks with a `tiled_camera` sensor, `_inject_camera_obs` replaces the
`policy` key in the obs dict with an `image` key:

- Picks the first available data type from `("rgb", "rgba", "depth",
  "distance_to_camera", "distance_to_image_plane")`.
- `rgba` is sliced to RGB; uint8 RGB tensors pass through.
- Depth / distance tensors are normalized to uint8 `[0, 255]` using the
  camera's configured `clipping_range[1]` as the far plane.

`IsaacLabVecEnv` then optionally resizes `image` to `image_size` via bilinear
interpolation on GPU, and exposes the renamed/resized space in
`observation_space`.

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDict


def create_isaaclab_env(config: Any) -> Any:
    """Create a patched IsaacLab gym env. AppLauncher must already be created."""
    import isaaclab_tasks  # noqa: F401  — registers gym entries
    from isaaclab_tasks.utils import parse_env_cfg

    # "isaaclab_Isaac-Cartpole-RGB-Camera-Direct-v0" → "Isaac-Cartpole-RGB-Camera-Direct-v0"
    gym_id = str(config.task).split("_", 1)[1]
    env_cfg = parse_env_cfg(gym_id, device=config.device, num_envs=int(config.env_num))
    env = gym.make(gym_id, cfg=env_cfg)
    _patch_env(env.unwrapped)

    return env


def _patch_env(env: Any) -> None:
    """Defer IsaacLab's auto-reset by one step to preserve terminal obs for WM training.

    Also injects tiled_camera images into the observation dict when available.
    """
    from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv

    if not isinstance(env, DirectRLEnv) and not isinstance(env, ManagerBasedRLEnv):
        raise TypeError(f"Unsupported IsaacLab env type: {type(env)}. Expected DirectRLEnv or ManagerBasedRLEnv.")

    has_camera = "tiled_camera" in env.scene.sensors
    original_cls = type(env)

    class _DeferredResetEnv(original_cls):
        _has_tiled_camera = has_camera

        # True only while super().step() is running; gates _reset_idx interception.
        _block_reset: bool = False

        # ---- core override: deferred reset logic --------------------------

        def _reset_idx(self, env_ids):
            # Block auto-resets triggered by the parent step() call so that terminal obs are preserved.
            if not self._block_reset:
                super()._reset_idx(env_ids)
                return

        def step(self, action: torch.Tensor, done: torch.Tensor | None = None):
            """Step with explicit reset control via the trainer's ``done`` flag.

            1. Run the parent step() with all auto-resets intercepted so that
               terminal observations are preserved.
            2. Explicitly reset envs marked ``done`` by the trainer (previous
               step's terminations) and overwrite their obs with initial obs.
            """
            # Zero actions for done envs so the unavoidable junk step on terminal state stays neutral.
            if done is not None:
                action = torch.where(done.unsqueeze(-1), torch.zeros_like(action), action)

            # Run parent step(); _reset_idx calls are intercepted above.
            self._block_reset = True
            result = super().step(action)
            self._block_reset = False

            # Explicitly reset envs that the trainer marked as done.
            if done is not None and done.any():
                reset_ids = done.nonzero(as_tuple=False).squeeze(-1)

                # Correct IsaacLab's internal flags — returned via result as terminated/truncated.
                self.reset_terminated[reset_ids] = False
                self.reset_time_outs[reset_ids] = False
                self.reset_buf[reset_ids] = False

                # Roll back the history append done by super().step()'s compute(update_history=True)
                # so the post-reset recompute below can refill reset envs via is_first_push without
                # double-appending for non-reset envs.
                # NOTE: touches CircularBuffer private fields (_pointer, _num_pushes,
                # _group_obs_term_history_buffer). Tied to IsaacLab 2.3.2 (pinned in
                # pyproject.toml). Re-validate this block when bumping IsaacLab.
                om = getattr(self, "observation_manager", None)
                if om is not None:
                    for group_buffers in om._group_obs_term_history_buffer.values():
                        for cb in group_buffers.values():
                            if cb._buffer is None:
                                continue
                            cb._pointer = (cb._pointer - 1) % cb.max_length
                            cb._num_pushes -= 1

                # Actually reset here.
                original_cls._reset_idx(self, reset_ids)
                if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                    for _ in range(self.cfg.num_rerenders_on_reset):
                        self.sim.render()

                # Refresh obs_buf with initial obs for reset envs. For ManagerBased envs we
                # re-run compute(update_history=True) so reset envs' history is filled via
                # is_first_push (matching default IsaacLab semantics).
                if om is not None:
                    # om.compute() returns a fresh dict, so rebuild result to point to it.
                    self.obs_buf = om.compute(update_history=True)
                    result = (self.obs_buf, *result[1:])
                else:
                    initial_obs = original_cls._get_observations(self)
                    for k, v in initial_obs.items():
                        self.obs_buf[k][reset_ids] = v[reset_ids]

            if self._has_tiled_camera:
                return (self._inject_camera_obs(result[0]), *result[1:])

            return result

        # ---- observation helpers ------------------------------------------

        # Preferred camera data types in priority order: color first, then depth-like.
        _CAMERA_DATA_TYPES = (
            "rgb",
            "rgba",
            "depth",
            "distance_to_camera",
            "distance_to_image_plane",
        )

        def _inject_camera_obs(self, obs):
            """Replace "policy" key with tiled_camera "image" for CNN encoder."""
            camera = self.scene.sensors["tiled_camera"]

            raw = None
            for data_type in self._CAMERA_DATA_TYPES:
                if data_type in camera.data.output:
                    raw = camera.data.output[data_type].clone()
                    break

            if raw is not None:
                if raw.shape[-1] == 4:
                    raw = raw[..., :3]  # RGBA → RGB
                raw = self._normalize_camera(raw)
                obs = dict(obs)
                obs.pop("policy", None)
                obs["image"] = raw
            return obs

        def _normalize_camera(self, raw: torch.Tensor) -> torch.Tensor:
            """Convert camera tensor to uint8 [0, 255].

            * RGB/RGBA — already uint8, returned as-is.
            * Depth / distance (float32) — normalized via the camera's clipping range.
            """
            if raw.dtype == torch.uint8:
                return raw
            # Depth/distance: normalize [0, clip_far] → [0, 255] using the configured clipping range.
            clip_far = self.scene.sensors["tiled_camera"].cfg.spawn.clipping_range[1]
            return (raw / clip_far * 255.0).clamp(0, 255).to(torch.uint8)

    _DeferredResetEnv.__name__ = f"DeferredReset_{original_cls.__name__}"
    _DeferredResetEnv.__qualname__ = _DeferredResetEnv.__name__
    env.__class__ = _DeferredResetEnv


def _resize_images(images: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    """Resize (N,H,W,C) batch on GPU via bilinear interpolation, preserving dtype."""
    src_dtype = images.dtype
    x = images.permute(0, 3, 1, 2).float()  # (N,C,H,W) for F.interpolate
    x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
    x = x.permute(0, 2, 3, 1)  # back to (N,H,W,C)
    if src_dtype == torch.uint8:
        return x.clamp(0, 255).to(torch.uint8)
    return x


class IsaacLabVecEnv:
    """ParallelEnv-compatible wrapper around a vectorized IsaacLab env (all tensors on GPU)."""

    def __init__(
        self,
        env: Any,
        simulation_app: Any | None = None,
        image_size: tuple[int, int] | None = None,
    ):
        self._env = env
        self._unwrapped_env = env.unwrapped
        self._app = simulation_app
        self._image_size = tuple(image_size) if image_size is not None else None
        self._num_envs: int = self._unwrapped_env.num_envs
        self._device: torch.device = self._unwrapped_env.device
        self._has_tiled_camera: bool = getattr(self._unwrapped_env, "_has_tiled_camera", False)

        # IsaacLab requires reset() before the first step().
        self._env.reset()

        # Build and cache spaces once (avoid re-creation on every property access).
        self._observation_space = self._build_observation_space()
        self._action_space = self._build_action_space()

    # --- ParallelEnv-compatible interface ---

    @property
    def env_num(self) -> int:
        return self._num_envs

    @property
    def observation_space(self) -> gym.spaces.Dict:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    def _build_observation_space(self) -> gym.spaces.Dict:
        """Single-env obs space with Dreamer flags. Camera envs: "policy" → "image" (resized)."""
        spaces: dict = {}
        for key, box in self._unwrapped_env.single_observation_space.spaces.items():
            spaces[key] = gym.spaces.Box(
                low=np.asarray(box.low, dtype=np.float32),
                high=np.asarray(box.high, dtype=np.float32),
                dtype=box.dtype,
            )

        # Vision patch renames "policy" → "image" at runtime; mirror that here.
        if self._has_tiled_camera and "policy" in spaces and "image" not in spaces:
            spaces["image"] = spaces.pop("policy")

        # Override image space with target size and uint8 dtype.
        if "image" in spaces:
            orig_shape = spaces["image"].shape
            if self._image_size is not None and len(orig_shape) == 3:
                h, w = self._image_size
                shape = (h, w, orig_shape[-1])
            else:
                shape = orig_shape
            spaces["image"] = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)

        return gym.spaces.Dict(spaces)

    def _build_action_space(self) -> gym.spaces.Box:
        """Single-env action space, always [-1, 1].

        IsaacLab declares unbounded spaces ([-inf, inf]) by convention;
        the task's ``_apply_action`` handles scaling to physical units internally.
        """
        shape = self._unwrapped_env.single_action_space.shape
        return gym.spaces.Box(-1.0, 1.0, shape=shape, dtype=np.float32)

    def _process_obs(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Fix dtypes, resize images, lift 1-D → (B,1)."""
        data: dict[str, torch.Tensor] = {}
        for key, val in obs_dict.items():
            if val.dtype == torch.float64:
                val = val.float()
            if key == "image" and self._image_size is not None and val.ndim == 4:
                h, w = self._image_size
                if val.shape[1] != h or val.shape[2] != w:
                    val = _resize_images(val, self._image_size)
            if val.ndim == 1:
                val = val.unsqueeze(-1)
            data[key] = val
        return data

    def step(self, action: torch.Tensor, done: torch.Tensor) -> tuple[TensorDict, torch.Tensor]:
        """Step all envs and return (TensorDict(B,), episode_done(B,)) on GPU.

        _DeferredResetEnv preserves terminal obs and resets ``done`` envs with
        initial obs. This wrapper sets ``is_first`` and zeros reward, mirroring ParallelEnv.
        """

        # Call the unwrapped (patched) env directly so that the ``done``
        # flag reaches ``_DeferredResetEnv.step()``.
        action = action.to(self._device)
        done = done.to(self._device)
        obs_dict, reward, terminated, truncated, extras = self._unwrapped_env.step(action, done=done)

        # _DeferredResetEnv.step already zeroed reset_terminated/reset_time_outs in-place for ``done`` envs.
        episode_done = terminated | truncated

        data = self._process_obs(obs_dict)

        # Trainer's ``done`` flags become ``is_first``; these envs were reset
        # by _DeferredResetEnv during this step(), so obs are already initial.
        data["is_first"] = done.unsqueeze(-1)
        data["is_terminal"] = terminated.unsqueeze(-1)
        data["is_last"] = episode_done.unsqueeze(-1)

        # Zero reward for is_first envs (no meaningful action produced this obs).
        reward_out = reward.float()
        if done.any():
            reward_out = torch.where(done, torch.zeros_like(reward_out), reward_out)
        data["reward"] = reward_out.unsqueeze(-1)

        # Forward log_* tensors from top-level extras (DirectRL) and extras["log"] sub-dict (ManagerBased).
        log_extras = extras.get("log", {}) if isinstance(extras, dict) else {}
        log_items = {k: v for k, v in extras.items() if k.startswith("log_")}
        log_items.update({f"log_{k}": v for k, v in log_extras.items()})
        for key, val in log_items.items():
            if not isinstance(val, torch.Tensor):
                continue
            if val.ndim == 0:
                val = val.expand(self._num_envs)
            if val.ndim == 1:
                val = val.unsqueeze(-1)
            data[key] = val.to(self._device)

        td = TensorDict(data, batch_size=(self._num_envs,), device=self._device)
        return td, episode_done

    def close(self) -> None:
        """Close env and optionally shut down SimulationApp."""
        if hasattr(self._env, "close"):
            self._env.close()
        if self._app is not None:
            self._app.close()

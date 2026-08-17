import atexit

from . import parallel, wrappers


def make_envs(config):
    suite = config.task.split("_", 1)[0]

    if suite == "isaaclab":
        return _make_isaaclab_envs(config)

    def env_constructor(idx, train=True):
        return lambda: make_env(config, idx, train=train)

    train_envs = parallel.ParallelEnv(lambda i: env_constructor(i, train=True),
                                      config.env_num, config.device)
    eval_envs = (
        parallel.ParallelEnv(lambda i: env_constructor(i, train=False),
                             config.eval_episode_num, config.device)
        if config.eval_episode_num > 0
        else None
    )
    obs_space = train_envs.observation_space
    act_space = train_envs.action_space

    # Old-PPO-style self-play: when the env config enables an opponent pool,
    # attach a main-process controller that adapts the per-candidate sampling
    # priorities from in-training winrates.  The trainer drives it via
    # ``train_envs.self_play`` (None for every non-self-play setup).
    if suite == "cr":
        pool_cfg = getattr(config, "opponent_pool", None)
        if pool_cfg is not None and bool(getattr(pool_cfg, "enabled", False)):
            from envs.cr_opponents import SelfPlayController

            train_envs.self_play = SelfPlayController(
                train_envs,
                update_every=getattr(pool_cfg, "update_every", 5e4),
                alpha=getattr(pool_cfg, "alpha", 0.3),
                floor=getattr(pool_cfg, "floor", 0.1),
                verbose=getattr(pool_cfg, "verbose", False),
            )

    return train_envs, eval_envs, obs_space, act_space


def _make_isaaclab_envs(config):
    if int(config.eval_episode_num) > 0:
        raise ValueError(
            "IsaacLab environments do not support separate eval envs yet. Set eval_episode_num: 0 in the env config."
        )

    # AppLauncher must be created before any isaaclab.* module is imported.
    from isaaclab.app import AppLauncher

    headless = getattr(config, "headless", True)
    launcher = AppLauncher(headless=headless, enable_cameras=True)
    sim_app = launcher.app

    from envs.isaaclab import IsaacLabVecEnv, create_isaaclab_env

    raw_env = create_isaaclab_env(config)

    train_envs = IsaacLabVecEnv(raw_env, simulation_app=sim_app, image_size=tuple(config.size))

    # Ensure the SimulationApp is shut down cleanly on process exit.
    atexit.register(train_envs.close)

    eval_envs = None  # Not supported yet for IsaacLab.

    obs_space = train_envs.observation_space
    act_space = train_envs.action_space
    return train_envs, eval_envs, obs_space, act_space


def make_env(config, id, train=True):
    suite, task = config.task.split("_", 1)
    if suite == "dmc":
        import envs.dmc as dmc

        env = dmc.DeepMindControl(task, config.action_repeat, config.size, seed=config.seed + id)
        env = wrappers.NormalizeActions(env)
    elif suite == "atari":
        import envs.atari as atari

        env = atari.Atari(
            task,
            config.action_repeat,
            config.size,
            gray=config.gray,
            noops=config.noops,
            lives=config.lives,
            sticky=config.sticky,
            actions=config.actions,
            length=config.time_limit,
            pooling=config.pooling,
            aggregate=config.aggregate,
            resize=config.resize,
            autostart=config.autostart,
            clip_reward=config.clip_reward,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "memorymaze":
        from envs.memorymaze import MemoryMaze

        env = MemoryMaze(task, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "crafter":
        import envs.crafter as crafter

        env = crafter.Crafter(task, config.size, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "metaworld":
        import envs.metaworld as metaworld

        env = metaworld.MetaWorld(
            task,
            config.action_repeat,
            config.size,
            config.camera,
            config.seed + id,
        )
    elif suite == "cr":
        # Clash Royale simulator (clash-royale-simulator). The action space is
        # already a flat multi one-hot Box (multi_discrete), no wrapper needed.
        # Absolutize the logdir *before* importing envs.cr: that import chdirs
        # into the simulator dir, which would corrupt relative paths.
        import os

        abs_logdir = ""
        if getattr(config, "logdir", None):
            abs_logdir = os.path.abspath(config.logdir)

        import envs.cr as cr

        pool_cfg = getattr(config, "opponent_pool", None)
        if train and pool_cfg is not None and bool(getattr(pool_cfg, "enabled", False)):
            from envs.cr_opponents import build_opponent_pool

            pool = build_opponent_pool(pool_cfg, logdir=abs_logdir,
                                       rank=id, base_seed=config.seed)
            env = cr.ClashRoyale(pool=pool, seed=config.seed + id, speed=config.speed)
        else:
            # Eval envs (train=False) always use the fixed `opponent` — the
            # stable ruler for episode/eval_score (mirrors the old PPO
            # BestWeightCallback's fixed crowd).
            env = cr.ClashRoyale(opponent=config.opponent, seed=config.seed + id, speed=config.speed)
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit // config.action_repeat)
    return wrappers.Dtype(env)

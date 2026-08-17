import atexit
import contextlib
import enum
import os
import sys
import time
import traceback
from functools import partial as bind

import numpy as np
import torch
from tensordict import TensorDict

import tools


class ParallelEnv:
    def __init__(self, constructor, env_num, device):
        self.envs = [Parallel(constructor(i), "process") for i in range(env_num)]
        self.device = device

    @property
    def observation_space(self):
        return self.envs[0].observation_space

    @property
    def action_space(self):
        return self.envs[0].action_space

    @property
    def env_num(self):
        return len(self.envs)

    def lift_dim(self, td):
        for key in td.keys():
            if td[key].ndim == 1:
                td[key] = td[key].unsqueeze(-1)
        return td

    def call_each(self, name, *args, **kwargs):
        """Call ``name(*args, **kwargs)`` on every underlying env (in its worker).

        Used by the self-play controller to pull per-episode results
        (``get_episode_results``) and push adaptive pool priorities
        (``set_pool_priorities``) into each worker's ``OpponentPool``.
        """
        # The Parallel proxy returns an unresolved Future; calling it again
        # resolves the worker's answer.
        return [getattr(e, name)(*args, **kwargs)() for e in self.envs]

    def step(self, action, done):
        """Step all environments.

        Notes
        -----
        This implementation intentionally steps the environment processes on CPU.
        The returned TensorDict is pinned in CPU memory so that the caller can
        transfer it to GPU asynchronously (H2D with non_blocking=True).

        ``action`` and ``done`` may arrive on any device (CPU or CUDA).
        They are moved to CPU internally before dispatching to workers.
        """
        # Ensure inputs are on CPU for worker processes.
        action_np = tools.to_np(action)  # handles any device via .detach().cpu().numpy()
        done = done.cpu() if done.is_cuda else done
        promise = [e.reset() if d else e.step(a) for e, a, d in zip(self.envs, action_np, done)]
        new_o, new_r, new_d = [], [], []
        for p, d in zip(promise, done):
            if d:
                new_o.append(p())
                new_r.append(0.0)
                new_d.append(False)
            else:
                o, r, d, _ = p()
                new_o.append(o)
                new_r.append(r)
                new_d.append(d)
        obs_stacked = {k: np.stack([o[k] for o in new_o]) for k in new_o[0].keys()}

        # Build CPU tensors first to avoid implicit GPU syncs and enable async H2D in caller.
        obs_tensors = {k: torch.as_tensor(v, device="cpu") for k, v in obs_stacked.items()}
        rew_stacked = torch.as_tensor(new_r, dtype=torch.float32, device="cpu")

        # Keep data on CPU; caller will .to(device, non_blocking=True) after pinning.
        # TensorDict batch size is (B,).
        td = TensorDict({**obs_tensors, "reward": rew_stacked}, batch_size=(self.env_num,), device="cpu")
        if torch.cuda.is_available():  # pin_memory requires an accelerator (CPU-only safe)
            td = td.pin_memory()
        done = torch.as_tensor(new_d, device="cpu")
        return self.lift_dim(td), done


class Parallel:
    def __init__(self, constructor, strategy):
        self.worker = Worker(bind(self._respond, constructor), strategy, state=True)
        self.callables = {}

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            if name not in self.callables:
                self.callables[name] = self.worker(PMessage.CALLABLE, name)()
            if self.callables[name]:
                return bind(self.worker, PMessage.CALL, name)
            return self.worker(PMessage.READ, name)()
        except AttributeError:
            raise ValueError(name)

    def __len__(self):
        return self.worker(PMessage.CALL, "__len__")()

    def close(self):
        self.worker.close()

    @staticmethod
    def _attr(state, name):
        """Resolve an attribute on the (possibly wrapped) env.

        Plain ``getattr`` only sees the outermost wrapper's own attributes; in
        gymnasium >= 1.1 the ``Wrapper`` chain is no longer transparent via
        ``__getattr__``.  Fall back to ``get_wrapper_attr`` so the self-play
        hooks (``get_episode_results`` / ``set_pool_priorities``) defined on the
        inner ``ClashRoyale`` env stay reachable from the main process.
        """
        try:
            return getattr(state, name)
        except AttributeError:
            get_wrapper_attr = getattr(state, "get_wrapper_attr", None)
            if get_wrapper_attr is not None:
                return get_wrapper_attr(name)
            raise

    @staticmethod
    def _respond(constructor, state, message, name, *args, **kwargs):
        state = state or constructor()  # Instantiate at first time
        if message == PMessage.CALLABLE:
            assert not args and not kwargs, (args, kwargs)
            result = callable(Parallel._attr(state, name))
        elif message == PMessage.CALL:
            result = Parallel._attr(state, name)(*args, **kwargs)
        elif message == PMessage.READ:
            assert not args and not kwargs, (args, kwargs)
            result = Parallel._attr(state, name)
        return state, result


class PMessage(enum.Enum):
    CALLABLE = 2
    CALL = 3
    READ = 4


class Worker:
    initializers = []

    def __init__(self, fn, strategy="thread", state=False):
        if not state:

            def fn_wrapper(s, *args, **kwargs):
                return (s, fn(*args, **kwargs))

            fn = fn_wrapper
        inits = self.initializers
        self.impl = {
            "process": bind(ProcessPipeWorker, initializers=inits),
            "daemon": bind(ProcessPipeWorker, initializers=inits, daemon=True),
        }[strategy](fn)
        self.promise = None

    def __call__(self, *args, **kwargs):
        self.promise and self.promise()  # Raise previous exception if any.
        self.promise = self.impl(*args, **kwargs)
        return self.promise

    def wait(self):
        return self.impl.wait()

    def close(self):
        self.impl.close()


class ProcessPipeWorker:
    def __init__(self, fn, initializers=(), daemon=False):
        import multiprocessing

        import cloudpickle

        self._context = multiprocessing.get_context("spawn")
        self._pipe, pipe = self._context.Pipe()
        fn = cloudpickle.dumps(fn)
        initializers = cloudpickle.dumps(initializers)
        self._process = self._context.Process(target=self._loop, args=(pipe, fn, initializers), daemon=daemon)
        self._process.start()
        self._nextid = 0
        self._results = {}
        assert self._submit(Message.OK)()
        atexit.register(self.close)

    def __call__(self, *args, **kwargs):
        return self._submit(Message.RUN, (args, kwargs))

    def wait(self):
        pass

    def close(self):
        try:
            self._pipe.send((Message.STOP, self._nextid, None))
            self._pipe.close()
        except (AttributeError, OSError):
            pass  # The connection was already closed.
        try:
            self._process.join(0.1)
            if self._process.exitcode is None:
                try:
                    os.kill(self._process.pid, 9)
                    time.sleep(0.1)
                except Exception:
                    pass
        except (AttributeError, AssertionError):
            pass

    def _submit(self, message, payload=None):
        callid = self._nextid
        self._nextid += 1
        self._pipe.send((message, callid, payload))
        return Future(self._receive, callid)

    def _receive(self, callid):
        while callid not in self._results:
            try:
                message, received_callid, payload = self._pipe.recv()
            except (OSError, EOFError):
                raise RuntimeError("Lost connection to worker.")
            if message == Message.ERROR:
                raise Exception(payload)
            if message == Message.RESULT:
                self._results[received_callid] = payload
            else:
                raise RuntimeError(f"Unexpected message: {message}")
        return self._results.pop(callid)

    @staticmethod
    def _loop(pipe, function, initializers):
        try:
            callid = None
            state = None
            import cloudpickle

            initializers = cloudpickle.loads(initializers)
            function = cloudpickle.loads(function)
            [fn() for fn in initializers]
            while True:
                if not pipe.poll(0.1):
                    continue  # Wake up for keyboard interrupts.
                message, callid, payload = pipe.recv()
                if message == Message.STOP:
                    return
                if message == Message.OK:
                    pipe.send((Message.RESULT, callid, True))
                elif message == Message.RUN:
                    args, kwargs = payload
                    state, result = function(state, *args, **kwargs)
                    pipe.send((Message.RESULT, callid, result))
                else:
                    raise KeyError(f"Invalid message: {message}")
        except (EOFError, KeyboardInterrupt):
            pass
        except Exception:
            stacktrace = "".join(traceback.format_exception(*sys.exc_info()))
            print(f"Error inside process worker: {stacktrace}.", flush=True)
            pipe.send((Message.ERROR, callid, stacktrace))
        finally:
            with contextlib.suppress(Exception):
                pipe.close()


class Message(enum.Enum):
    OK = 1
    RUN = 2
    RESULT = 3
    STOP = 4
    ERROR = 5


class Future:
    def __init__(self, receive, callid):
        self._receive = receive
        self._callid = callid
        self._result = None
        self._complete = False

    def __call__(self):
        if not self._complete:
            self._result = self._receive(self._callid)
            self._complete = True
        return self._result

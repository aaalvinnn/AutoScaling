import os

import gymnasium as gym


DEFAULT_VECTOR_BACKEND = "spawn"
VALID_VECTOR_BACKENDS = {"sync", "fork", "spawn", "forkserver"}


def make_vector_env(env_fns, default_backend=DEFAULT_VECTOR_BACKEND):
    """Create a Gymnasium vector env with a configurable multiprocessing backend."""
    backend = os.environ.get("AUTOSCALING_VECTOR_BACKEND", default_backend).lower()
    if backend not in VALID_VECTOR_BACKENDS:
        raise ValueError(
            f"Unknown AUTOSCALING_VECTOR_BACKEND={backend!r}; "
            f"expected one of {sorted(VALID_VECTOR_BACKENDS)}"
        )

    if backend == "sync":
        print("VectorEnv backend: sync")
        return gym.vector.SyncVectorEnv(env_fns)

    shared_memory = os.environ.get("AUTOSCALING_VECTOR_SHARED_MEMORY", "1") != "0"
    print(f"VectorEnv backend: async context={backend} shared_memory={shared_memory}")
    return gym.vector.AsyncVectorEnv(
        env_fns,
        context=backend,
        shared_memory=shared_memory,
    )

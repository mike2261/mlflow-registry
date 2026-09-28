"""MLflow ``python_function`` adapters that serve the registered speech models.

Each registered model (raw Hugging Face snapshot, version 1) gets a thin pyfunc
wrapper (version 2, alias ``serving``) that downloads the snapshot at load time
and runs it with the model's own runtime. ``mlflow models serve`` then exposes
the uniform REST contract described in ``base.py``.

Runtime packages (torch, transformers, nemo, ...) are imported lazily inside
``load_context`` so this package itself only depends on mlflow, numpy, soundfile
and scipy. That lets the registration script run anywhere while the heavy
imports happen only inside the per-model Docker image.
"""

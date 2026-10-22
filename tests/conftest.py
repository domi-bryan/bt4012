# Load torch before anything imports LightGBM. On macOS both ship their own OpenMP runtime,
# and training a torch model after LightGBM was imported segfaults; the reverse order is safe.
import torch  # noqa: F401

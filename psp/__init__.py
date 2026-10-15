"""psp: parameter space partitioning MCMC for global model evaluation, in PyTorch.

>>> from psp import PSP
"""

from ._callbacks import Discretize, Model, rowwise
from ._control import UNBOUNDED, Control
from ._global import PSP, PSPResult, clamp_parameters, hyper_points
from ._io import SavedRun, has_pyarrow, read_saved, saved_paths
from ._many import ManyModel, pspMany

__version__ = "0.1.0"

__all__ = [
    "PSP",
    "pspMany",
    "ManyModel",
    "PSPResult",
    "Control",
    "UNBOUNDED",
    "rowwise",
    "Model",
    "Discretize",
    "hyper_points",
    "clamp_parameters",
    "read_saved",
    "saved_paths",
    "has_pyarrow",
    "SavedRun",
    "__version__",
]

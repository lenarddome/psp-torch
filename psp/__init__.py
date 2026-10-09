"""psp: parameter space partitioning MCMC for global model evaluation, in PyTorch.

A port of the R package `psp <https://lenarddome.com/psp/>`_ (Dome & Wills) that
keeps its interface: :func:`pspGlobal` takes a ``model``, a ``discretize``
function and a ``control`` mapping, and returns the ordinal patterns it found
with their populations. Models are evaluated in batches, so a run can use the GPU.

>>> from psp import pspGlobal
"""

from ._callbacks import Discretize, Model, rowwise
from ._control import UNBOUNDED, Control
from ._global import PSPResult, clamp_parameters, hyper_points, pspGlobal
from ._io import SavedRun, has_pyarrow, read_saved, saved_paths
from ._many import ManyModel, pspMany

__version__ = "0.1.0"

__all__ = [
    "pspGlobal",
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

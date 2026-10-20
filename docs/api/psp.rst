``psp``
=======

.. module:: psp

The functions you will use to run a search and read its results. Everything else the package
exports is under `Advanced`_.

.. autosummary::

   PSP
   PSPLockstep
   PSPResult
   rowwise
   read_saved

Running a search
----------------

.. autofunction:: PSP

.. autoclass:: PSPResult
   :members:

.. autofunction:: PSPLockstep

Models written for one parameter set
------------------------------------

.. autofunction:: rowwise

Reading saved runs
------------------

.. autofunction:: read_saved

Advanced
--------

The control settings as a class, the types of the callbacks, the pieces of the saved-run reader,
and the two steps of a proposal. You need none of them to run a search.

.. autosummary::

   Control
   UNBOUNDED
   Model
   Discretize
   LockstepModel
   SavedRun
   saved_paths
   has_pyarrow
   hyper_points
   clamp_parameters

.. autoclass:: Control
   :members:

.. autodata:: UNBOUNDED

.. autodata:: Model

.. autodata:: Discretize

.. autodata:: LockstepModel

.. autoclass:: SavedRun
   :members:

.. autofunction:: saved_paths

.. autofunction:: has_pyarrow

.. autofunction:: hyper_points

.. autofunction:: clamp_parameters

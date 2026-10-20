"""The store of ordinal patterns found by a run: their matrices, populations and centres."""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["PatternStore"]


class PatternStore:
    """Bookkeeping utility. Stores every ordinal pattern found so far,
    their frequencies, and the centres of the jumping distributions (proposes
    candidate parameter sets; the points in the parameter space used to jump
    from to new points).

    Two matrices are the same pattern when they are identical element by element,
    with ``-0.0`` treated as ``0.0`` and every ``NaN`` treated as the same ``NaN``.
    Patterns are numbered in the order they were found, from 0.

    Parameters
    ----------
    dimensions
        Number of parameters, the width of a centre.
    device, dtype
        Where and how the centres are kept; they live with the sampler.
    """

    def __init__(self, dimensions: int, device: torch.device, dtype: torch.dtype) -> None:
        self._index: dict[bytes, int] = {}
        self._patterns: list[torch.Tensor] = []
        self._counts = torch.zeros(0, dtype=torch.int64)
        self._centres = torch.empty((0, dimensions), device=device, dtype=dtype)

    def __len__(self) -> int:
        return len(self._patterns)

    @property
    def counts(self) -> torch.Tensor:
        """Population of each pattern, shape ``(patterns,)``, on the CPU."""
        return self._counts

    @property
    def centres(self) -> torch.Tensor:
        """The last parameter set that produced each pattern, shape ``(patterns, dimensions)``."""
        return self._centres

    def patterns(self) -> torch.Tensor:
        """The patterns found, shape ``(patterns, dimensionality, dimensionality)``, float64."""
        if not self._patterns:
            return torch.empty((0, 0, 0), dtype=torch.float64)
        return torch.stack(self._patterns)

    def update(self, ordinal: torch.Tensor, jumping: torch.Tensor) -> torch.Tensor:
        """Register one evaluation per row and return the pattern id of each.

        Parameters
        ----------
        ordinal
            Patterns produced, shape ``(n, dimensionality, dimensionality)``.
        jumping
            The parameter sets that produced them, shape ``(n, dimensions)``.

        Returns
        -------
        torch.Tensor
            Pattern ids, shape ``(n,)``, ``int64``, on the CPU.
        """
        n = int(ordinal.shape[0])
        if n == 0:
            return torch.empty(0, dtype=torch.int64)
        ## canonical bytes: -0 folded into 0 and every NaN folded into the same NaN
        ## move catalogued ordinal patterns to CPU and convert to float64 for comparison
        canonical = ordinal.detach().to("cpu", torch.float64).numpy() + 0.0
        canonical[np.isnan(canonical)] = np.nan
        ids = np.empty(n, dtype=np.int64)
        new_rows: list[int] = []
        ## compare byte-representations of each stored pattern to the new ones, and add
        ## any new patterns to the store
        for i in range(n):
            key = canonical[i].tobytes()
            pattern_id = self._index.get(key)
            if pattern_id is None:
                pattern_id = len(self._patterns)
                self._index[key] = pattern_id
                self._patterns.append(torch.from_numpy(canonical[i].copy()))
                new_rows.append(i)
            ids[i] = pattern_id
        ids_tensor = torch.from_numpy(ids)

        ## adds new rows to the counts and centres tensors, and
        ## updates the counts and centres of all patterns
        if new_rows:
            grown = len(self._patterns)
            self._counts = torch.cat(
                [self._counts, torch.zeros(grown - self._counts.numel(), dtype=torch.int64)]
            )
            self._centres = torch.cat(
                [
                    self._centres,
                    torch.empty(
                        (grown - self._centres.shape[0], self._centres.shape[1]),
                        device=self._centres.device,
                        dtype=self._centres.dtype,
                    ),
                ]
            )
        self._counts += torch.bincount(ids_tensor, minlength=len(self._patterns))

        ## each pattern's centre is the last parameter set in this batch that produced it
        device_ids = ids_tensor.to(self._centres.device)
        last = torch.full(
            (len(self._patterns),), -1, dtype=torch.int64, device=self._centres.device
        )
        last.scatter_reduce_(
            0, device_ids, torch.arange(n, device=self._centres.device), reduce="amax"
        )
        touched = last >= 0
        self._centres[touched] = jumping[last[touched]].to(self._centres.dtype)
        return ids_tensor

    def underpopulated(self, population: int) -> torch.Tensor:
        """Ids of the patterns with fewer than ``population`` parameter sets, shape ``(k,)``."""
        return torch.nonzero(self._counts < population, as_tuple=False).reshape(-1)

    def centres_of(self, ids: torch.Tensor) -> torch.Tensor:
        """The centres of the given pattern ids, shape ``(len(ids), dimensions)``."""
        return self._centres[ids.to(self._centres.device)]

# -*- coding: utf-8 -*-
"""Optional acceleration backends (torch_cluster, torch_scatter).

This module centralizes optional imports so other modules can rely on a single
source of truth for availability flags and functions.
"""

from __future__ import annotations

from typing import Any, Tuple

import torch

# Optional fast kNN backend
_HAS_TORCH_CLUSTER = False
try:
    from torch_cluster import knn_graph, knn  # type: ignore
    _HAS_TORCH_CLUSTER = True
except Exception:  # pragma: no cover
    knn_graph = None  # type: ignore
    knn = None  # type: ignore
    _HAS_TORCH_CLUSTER = False

# Optional scatter backend (used for fast segment softmax and some graph ops)
_HAS_SCATTER = False
_HAS_SCATTER_MIN = False
try:
    from torch_scatter import scatter_max, scatter_sum, scatter_min  # type: ignore
    _HAS_SCATTER = True
    _HAS_SCATTER_MIN = True
except Exception:  # pragma: no cover
    try:
        from torch_scatter import scatter_max, scatter_sum  # type: ignore
        scatter_min = None  # type: ignore
        _HAS_SCATTER = True
        _HAS_SCATTER_MIN = False
    except Exception:  # pragma: no cover
        scatter_max = None  # type: ignore
        scatter_sum = None  # type: ignore
        scatter_min = None  # type: ignore
        _HAS_SCATTER = False
        _HAS_SCATTER_MIN = False

__all__ = [
    "_HAS_TORCH_CLUSTER",
    "knn_graph",
    "knn",
    "_HAS_SCATTER",
    "_HAS_SCATTER_MIN",
    "scatter_max",
    "scatter_sum",
    "scatter_min",
]

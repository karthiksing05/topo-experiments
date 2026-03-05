"""
toposeed — Topographic seed-based neural network layers.
"""

from .layer import TopoSeedLayer
from .sheet import CorticalSheet
from .buffers import EvidenceBuffer, HealthBuffer, ResidualBuffer
from .expansion import ExpansionManager
from .regularizers import intra_layer_smoothness, cross_layer_coherence
from .utils import (
    compute_sheet_dimensions,
    place_seeds_on_grid,
    spatial_distance_matrix,
    gaussian_blur_2d,
    get_device,
)
from .visualize import (
    plot_cortical_sheet_weights,
    plot_active_mask,
    plot_selectivity_maps,
    plot_multi_layer_selectivity,
    plot_all,
)

__all__ = [
    "TopoSeedLayer",
    "CorticalSheet",
    "EvidenceBuffer",
    "HealthBuffer",
    "ResidualBuffer",
    "ExpansionManager",
    "intra_layer_smoothness",
    "cross_layer_coherence",
    "compute_sheet_dimensions",
    "place_seeds_on_grid",
    "spatial_distance_matrix",
    "gaussian_blur_2d",
    "get_device",
    "plot_cortical_sheet_weights",
    "plot_active_mask",
    "plot_selectivity_maps",
    "plot_multi_layer_selectivity",
    "plot_all",
]

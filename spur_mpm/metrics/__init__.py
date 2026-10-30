"""Image, vessel-structure and statistics metrics.

Three families, deliberately separate:

* ``image_metrics`` -- intensity agreement (MAE / PSNR / SSIM) and the
  ``split_errors`` that separate measured rows from missing rows. Intensity
  metrics are dominated by the dense, bright background; they are reported but
  never used alone to claim that anatomy was recovered.
* ``vessel`` / ``skeleton`` -- structural agreement after binarisation, where the
  threshold is fitted on the *reference* only. ``skeleton`` measures what
  survives thinning (length, branches, junctions, endpoints, radius), which is
  the ImageJ-comparable verdict on whether the vascular *network* was rebuilt.
* ``complexity`` -- parameters, MACs and measured latency. Kept here so a
  comparison table can quote what an arm costs next to what it scores.
"""

from spur_mpm.metrics.complexity import (
    complexity_report,
    count_macs,
    count_parameters,
    measure_latency,
    throughput_for_volume,
)
from spur_mpm.metrics.image_metrics import mae, psnr, split_errors, ssim
from spur_mpm.metrics.skeleton import analyse_volume, skeleton_overlap, skeleton_stats
from spur_mpm.metrics.statistics import bootstrap_ci, paired_delta, summarise_by_group
from spur_mpm.metrics.vessel import cl_dice, dice, vessel_mask, vessel_metrics

__all__ = ["mae", "psnr", "split_errors", "ssim",
           "bootstrap_ci", "paired_delta", "summarise_by_group",
           "cl_dice", "dice", "vessel_mask", "vessel_metrics",
           "analyse_volume", "skeleton_overlap", "skeleton_stats",
           "complexity_report", "count_macs", "count_parameters",
           "measure_latency", "throughput_for_volume"]

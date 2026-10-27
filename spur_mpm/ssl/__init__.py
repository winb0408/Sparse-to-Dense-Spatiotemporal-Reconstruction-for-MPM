"""Self-supervision: measurement holdout, MMS objective, leakage guards."""

from spur_mpm.ssl.holdout import HOLDOUT_GEOMETRIES, HoldoutSpec, split_holdout
from spur_mpm.ssl.leakage import (
    LeakageError,
    assert_holdout_is_measured,
    assert_inputs_blind,
    assert_split_disjoint,
)
from spur_mpm.ssl.mms import MMSLoss, charbonnier

__all__ = [
    "HOLDOUT_GEOMETRIES", "HoldoutSpec", "split_holdout",
    "MMSLoss", "charbonnier",
    "LeakageError", "assert_holdout_is_measured", "assert_inputs_blind",
    "assert_split_disjoint",
]

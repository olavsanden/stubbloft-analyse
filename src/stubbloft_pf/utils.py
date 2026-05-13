# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Union

import numpy as np

try:
    trapz = np.trapezoid
except AttributeError:
    trapz = np.trapz

def trap_integral(y: np.ndarray, x: np.ndarray, axis: int = -1) -> Union[float, np.ndarray]:
    # Wrapper rundt numpy sin trapesintegrasjon.

    return trapz(y, x, axis=axis)

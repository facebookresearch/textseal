# Copyright (c) Meta Platforms, Inc. and affiliates.
# DEPRECATED: Use textseal.common.config instead

import warnings
from textseal.common.utils.config import *

warnings.warn(
    "textseal.wmtraining.lingua.args is deprecated. Use textseal.common.config instead.",
    DeprecationWarning,
    stacklevel=2
)
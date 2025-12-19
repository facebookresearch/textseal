# Copyright (c) Meta Platforms, Inc. and affiliates.
# DEPRECATED: Use apps.common.config instead

import warnings
from apps.common.utils.config import *

warnings.warn(
    "apps.wmtraining.lingua.args is deprecated. Use apps.common.config instead.",
    DeprecationWarning,
    stacklevel=2
)
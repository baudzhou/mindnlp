# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
OPT Models init
"""
from . import opt, opt_config, opt_tokenizer
from .opt import *
from .opt_config import *
from .opt_tokenizer import *

__all__ = []
__all__.extend(opt.__all__)
__all__.extend(opt_config.__all__)
__all__.extend(opt_tokenizer.__all__)

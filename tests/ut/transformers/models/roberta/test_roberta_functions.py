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
"""Test Bert functions"""
import pytest

from mindnlp.transformers import RobertaForMaskedLM

@pytest.mark.download
def test_resize_embed():
    """test from pretrained"""
    model = RobertaForMaskedLM.from_pretrained('roberta-base')
    assert model.roberta.embeddings.word_embeddings.vocab_size == model.config.vocab_size
    num_tokens = model.config.vocab_size
    model.resize_token_embeddings(num_tokens + 1)
    assert model.roberta.embeddings.word_embeddings.vocab_size == num_tokens + 1

# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .kv_cache import initialize_past_key_values
from .medusa_choices import mc_sim_7b_63

from .utils import(
    generate_medusa_buffers,
    reset_medusa_mode,
    initialize_medusa,
    generate_candidates,
    tree_decoding,
    evaluate_posterior,
    update_inference_inputs,
)


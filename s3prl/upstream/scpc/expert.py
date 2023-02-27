# Copyright 2023 Sean Robertson
#
# Much of this code is based on that of github.com/facebookresearch/cpc_audio, which is
# MIT-licensed. See LICENSE_cpc_audio for license details.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List

import torch
from .modules import Encoder


class UpstreamExpert(torch.nn.Module):
    encoder: Encoder

    def __init__(self, ckpt: str, **kwargs):
        super().__init__(**kwargs)
        self.name = "[scpc]"
        self.encoder = Encoder.from_checkpoint(ckpt, "cpu")

    def get_downsample_rates(self, key: str) -> int:
        return self.encoder.downsampling_factor

    def forward(self, wavs: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        if len(wavs) == 0:
            return {"hidden_states": torch.empty(0, 0, self.encoder.output_size)}
        lens = torch.tensor([w.numel() for w in wavs]).to(wavs[0].device)
        x = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True).unsqueeze(-1)
        x, lens = self.encoder(x, lens)
        return {"hidden_states": x}

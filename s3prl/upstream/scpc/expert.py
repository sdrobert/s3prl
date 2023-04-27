# Copyright 2023 Sean Robertson
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

import math
import json

from typing import Dict, List, Optional

import torch

from .modules import Encoder


class UpstreamExpert(torch.nn.Module):
    encoder: Encoder

    def __init__(self, ckpt: str, model_config: Optional[str] = None, **kwargs):
        super().__init__()
        if model_config is not None:
            try:
                from pydrobert.speech.alias import alias_factory_subclass_from_arg
                from pydrobert.speech.compute import FrameComputer
            except ImportError:
                raise ImportError(
                    "config file specified for scpc upstream model, but config is for "
                    "the feature frontend"
                )
            with open(model_config) as f:
                model_config = json.load(f)
            self.feat_extractor = alias_factory_subclass_from_arg(
                FrameComputer, model_config
            )
        else:
            self.feat_extractor = None
        self.name = "[scpc]"
        self.encoder = Encoder.from_checkpoint(ckpt, "cpu")

    def get_downsample_rates(self, key: str) -> int:
        rate = self.encoder.downsampling_factor
        if self.feat_extractor is not None:
            rate *= math.round(
                self.feat_extractor.sampling_rate
                * self.feat_extractor.frame_shift_ms
                / 1000
            )
        return rate

    def forward(self, wavs: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        if len(wavs) == 0:
            return {"hidden_states": torch.empty(0, 0, self.encoder.output_size)}
        if self.feat_extractor is not None:
            wavs = [
                torch.tensor(
                    self.feat_extractor.compute_full(w.flatten().numpy()),
                    device=w.device,
                )
                for w in wavs
            ]
        lens = torch.tensor([w.size(0) for w in wavs]).to(wavs[0].device)
        x = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True).unsqueeze(-1)
        x, lens = self.encoder(x, lens)
        return {"hidden_states": x}

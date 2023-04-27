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

import json
import warnings

from typing import Dict, List, Optional

import torch

from .modules import Encoder


def get_feature_extractor(config):
    # FIXME(sdrobert): support GPU comps in pydrobert-speech
    try:
        from pydrobert.speech.alias import alias_factory_subclass_from_arg
        from pydrobert.speech.compute import (
            FrameComputer,
            ShortTimeFourierTransformFrameComputer,
        )
        from pydrobert.speech.filters import Fbank
        from pydrobert.speech.config import LOG_FLOOR_VALUE
    except ImportError:
        raise ImportError(
            "feature config file specified for scpc upstream model, but "
            "pydrobert-speech is not installed"
        )
    with open(config) as f:
        config = json.load(f)
    base_extractor = alias_factory_subclass_from_arg(FrameComputer, config)
    ds_rate = round(base_extractor.sampling_rate * base_extractor.frame_shift_ms / 1000)

    def _extractor(wav: torch.Tensor) -> torch.Tensor:
        warnings.warn("Can't convert to torchaudio feature front-end. Will be slow")
        return torch.tensor(
            base_extractor.compute_full(wav.flatten().cpu().numpy()),
            device=wav.device,
        )

    if isinstance(
        base_extractor, ShortTimeFourierTransformFrameComputer
    ) and isinstance(base_extractor.bank, Fbank):
        from torchaudio.compliance.kaldi import fbank

        if "window_function" in config:
            if isinstance(config["window_function"], dict):
                window_type = config["window_function"]["name"]
            else:
                window_type = config["window_function"]
            assert isinstance(window_type, str)
            if window_type == "black":
                window_type = "blackman"
            elif window_type == "hann":
                window_type = "hanning"
            if window_type not in {"blackman", "hamming", "hanning"}:
                return _extractor, ds_rate
        else:
            window_type = "hanning"

        kwargs = dict(
            energy_floor=LOG_FLOOR_VALUE,
            frame_length=base_extractor.frame_length_ms,
            frame_shift=base_extractor.frame_shift_ms,
            high_freq=base_extractor.bank.supports_hz[-1][1],
            low_freq=base_extractor.bank.supports_hz[0][0],
            num_mel_bins=base_extractor.bank.num_filts,
            use_energy=base_extractor.includes_energy,
            sample_frequency=base_extractor.sampling_rate,
            use_log_fbank=base_extractor._log,  # FIXME(sdrobert): expose this!
            use_power=base_extractor._power,  # FIXME(sdrobert): this, too!
            window_type=window_type,
        )

        def _extractor(wav: torch.Tensor) -> torch.Tensor:
            return fbank(wav.unsqueeze(0), **kwargs)

    return _extractor, ds_rate


class UpstreamExpert(torch.nn.Module):
    encoder: Encoder

    def __init__(self, ckpt: str, model_config: Optional[str] = None, **kwargs):
        super().__init__()
        if model_config is not None:
            self.feat_extractor, self._feds_rate = get_feature_extractor(model_config)
        else:
            self.feat_extractor, self._feds_rate = None, 1
        self.name = "[scpc]"
        self.encoder = Encoder.from_checkpoint(ckpt, "cpu")

    def get_downsample_rates(self, key: str) -> int:
        rate = self.encoder.downsampling_factor
        if self.feat_extractor is not None:
            rate *= self._feds_rate
        return rate

    def forward(self, wavs: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        if len(wavs) == 0:
            return {"hidden_states": torch.empty(0, 0, self.encoder.output_size)}
        if self.feat_extractor is not None:
            wavs = [self.feat_extractor(w) for w in wavs]
        else:
            wavs = [w.unsqueeze(-1) for w in wavs]
        lens = torch.tensor([w.size(0) for w in wavs]).to(wavs[0].device)
        x = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True)
        x, lens = self.encoder(x, lens)
        return {"hidden_states": x}

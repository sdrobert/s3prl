import os

from .expert import UpstreamExpert as _UpstreamExpert
from s3prl.util.download import _urls_to_filepaths


def scpc_local(ckpt, **kwargs):
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, **kwargs)


def scpc_url(ckpt, refresh=False, **kwargs):
    return scpc_local(_urls_to_filepaths(ckpt, refresh=refresh), **kwargs)

"""
Copyright 2025 Apple Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


# Dependencies.
import pytest

import torch

import parameterized_transforms.transforms as ptx
import parameterized_transforms.core as ptc

import numpy as np

from itertools import product

import typing as t


@pytest.mark.parametrize(
    "size, p, scale, ratio, value, default_params_mode, channels, tx_mode",
    list(
        product(
            [
                (
                    224,
                    224,
                ),
            ],
            # p
            [0.0, 0.5, 1.0],
            # scale
            [
                (0.02, 0.33),
            ],
            # ratio
            [
                (3.0 / 4, 4.0 / 3),
            ],
            # value
            ["random", 0, 23, 255.0],
            # identity params mode
            [
                ptc.DefaultParamsMode.RANDOMIZED,
                ptc.DefaultParamsMode.UNIQUE,
            ],
            # channels
            [
                (3,),
                (4,),
            ],
            ["CASCADE", "CONSUME"],
        )
    ),
)
def test_tx_on_torch_tensors(
    size: t.Tuple[int],
    p: float,
    scale: t.Union[t.Tuple[float, float], t.List[float]],
    ratio: t.Union[t.Tuple[float, float], t.List[float]],
    value: t.Union[str, int, str, t.List[int], t.Tuple[int, int, int]],
    default_params_mode: ptc.DefaultParamsMode,
    channels: t.Tuple[int],
    tx_mode: str,
) -> None:

    # PIL-support is NOT guaranteed in torchvision

    if channels == (None,):
        channels = ()

    size = channels + size

    tx = ptx.RandomErasing(
        p=p,
        scale=scale,
        ratio=ratio,
        value=value,
        tx_mode=tx_mode,
        default_params_mode=default_params_mode,
    )

    img = torch.from_numpy(np.random.uniform(low=0, high=1.0, size=size))

    orig_params = tuple([3.0, 4.0, -1.0, 0.0])

    aug_1, params_1 = tx.cascade_transform(img, orig_params)
    assert len(params_1) - len(orig_params) == tx.param_count

    orig_params = ()
    aug_2, params_2 = tx.cascade_transform(img, orig_params)
    aug_3, params_3 = tx.consume_transform(img, params_2)

    assert orig_params == params_3

    if value != "random":
        assert torch.all(torch.eq(aug_2, aug_3))

    assert all([isinstance(elt, float) or isinstance(elt, int) for elt in params_2])

    id_params = tx.get_default_params(img=img, processed=True)
    id_aug_img, id_aug_rem_params = tx.consume_transform(img=img, params=id_params)
    if value != "random":
        assert id_aug_rem_params == ()
        assert torch.all(torch.eq(id_aug_img, img))


# Main.
if __name__ == "__main__":
    pass

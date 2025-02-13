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

import numpy as np

import PIL.Image as Image
import PIL.ImageChops as ImageChops

from itertools import product

import typing as t


@pytest.mark.parametrize(
    "size, to_size, channels, tx_mode",
    list(
        product(
            [
                (
                    224,
                    224,
                ),
                (
                    32,
                    32,
                ),
            ],
            [
                # 224,  # This gives error.
                28,
                24,
                3,
                8,
                [24, 24],
                [3, 3],
                [8, 8],
                (24, 24),
                (3, 3),
                (8, 8),
                [24, 3],
                [3, 8],
                (8, 24),
            ],
            [(3,), (4,), (None,)],
            ["CASCADE", "CONSUME"],
        )
    ),
)
def test_tx_on_PIL_images(
    size: t.Tuple[int],
    to_size: t.Union[int, t.List[int], t.Tuple[int, int]],
    channels: t.Tuple[int],
    tx_mode: str,
) -> None:
    if channels == (None,):
        channels = ()

    size = size + channels

    tx = ptx.FiveCrop(
        size=to_size,
        tx_mode=tx_mode,
    )

    img = Image.fromarray(
        np.random.randint(low=0, high=256, size=size).astype(np.uint8)
    )

    orig_params = tuple([3.0, 4.0, -1.0, 0.0])

    aug_1, params_1 = tx.cascade_transform(img, orig_params)
    assert len(params_1) - len(orig_params) == tx.param_count

    orig_params = ()
    aug_2, params_2 = tx.cascade_transform(img, orig_params)
    aug_3, params_3 = tx.consume_transform(img, params_2)

    assert orig_params == params_3
    assert all(
        [
            not ImageChops.difference(aug_2_component, aug_3_component).getbbox()
            for aug_2_component, aug_3_component in zip(aug_2, aug_3)
        ]
    )
    assert all([isinstance(elt, float) or isinstance(elt, int) for elt in params_2])

    id_params = tx.get_default_params(img=img, processed=True)
    id_aug_img, id_aug_rem_params = tx.consume_transform(img=img, params=id_params)
    if img.size == id_aug_img[0].size:
        assert all(
            not ImageChops.difference(img, an_id_aug_img).getbbox()
            for an_id_aug_img in id_aug_img
        )
        assert id_aug_rem_params == ()


@pytest.mark.parametrize(
    "size, to_size, channels, tx_mode",
    list(
        product(
            [
                (
                    224,
                    224,
                ),
                (
                    32,
                    32,
                ),
            ],
            [
                # 224,  # This gives error as crop size > image size is possible
                28,
                24,
                3,
                8,
                [24, 24],
                [3, 3],
                [8, 8],
                (24, 24),
                (3, 3),
                (8, 8),
                [24, 3],
                [3, 8],
                (8, 24),
            ],
            [(3,), (4,), (None,)],
            ["CASCADE", "CONSUME"],
        )
    ),
)
def test_tx_on_torch_tensors(
    size: t.Tuple[int],
    to_size: t.Union[int, t.List[int], t.Tuple[int, int]],
    channels: t.Tuple[int],
    tx_mode: str,
) -> None:
    if channels == (None,):
        channels = ()

    size = channels + size

    tx = ptx.FiveCrop(
        size=to_size,
        tx_mode=tx_mode,
    )

    img = torch.from_numpy(np.random.uniform(low=0, high=1.0, size=size))

    orig_params = tuple([3.0, 4.0, -1.0, 0.0])

    aug_1, params_1 = tx.cascade_transform(img, orig_params)
    assert len(params_1) - len(orig_params) == tx.param_count

    orig_params = ()
    aug_2, params_2 = tx.cascade_transform(img, orig_params)
    aug_3, params_3 = tx.consume_transform(img, params_2)

    assert orig_params == params_3
    assert all(
        [
            torch.all(torch.eq(aug_2_component, aug_3_component))
            for aug_2_component, aug_3_component in zip(aug_2, aug_3)
        ]
    )
    assert all([isinstance(elt, float) or isinstance(elt, int) for elt in params_2])

    id_params = tx.get_default_params(img=img, processed=True)
    id_aug_img, id_aug_rem_params = tx.consume_transform(img=img, params=id_params)
    if img.shape == id_aug_img[0].shape:
        assert all(
            torch.all(torch.eq(img, an_id_aug_img)) for an_id_aug_img in id_aug_img
        )
        assert id_aug_rem_params == ()


# Main.
if __name__ == "__main__":
    pass

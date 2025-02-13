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

from itertools import product

import typing as t


@pytest.mark.parametrize(
    "size, channels, tx_mode",
    list(
        product(
            [
                (
                    224,
                    224,
                ),
            ],
            [
                (None,),
                (3,),
                (4,),
            ],
            ["CASCADE", "CONSUME"],
        )
    ),
)
def test_tx_on_PIL_images_1(
    size: t.Tuple[int],
    channels: t.Tuple[int],
    tx_mode: str,
) -> None:
    if channels == (None,):
        channels = ()

    size = size + channels

    tx = ptx.ToTensor()

    img = Image.fromarray(
        np.random.randint(low=0, high=256, size=size).astype(np.uint8)
    )

    orig_params = tuple([3.0, 4.0, -1.0, 0.0])

    aug_1, params_1 = tx(img, orig_params)

    aug_2, params_2 = tx.cascade_transform(img, orig_params)

    aug_3, params_3 = tx.consume_transform(img, orig_params)

    assert len(list(aug_1.shape)) == 3  # [C, H, W] format ensured
    assert aug_1.shape[0] == 1 or aug_1.shape[0] == 3 or aug_1.shape[0] == 4
    assert 0 <= torch.min(aug_1).item() <= 1.0
    assert 0 <= torch.max(aug_1).item() <= 1.0

    assert orig_params == params_1
    assert orig_params == params_2
    assert orig_params == params_3

    assert torch.all(torch.eq(aug_1, aug_2))
    assert torch.all(torch.eq(aug_1, aug_3))


@pytest.mark.parametrize(
    "size, channels, tx_mode",
    list(
        product(
            [
                (
                    224,
                    224,
                ),
            ],
            [
                (1,),
            ],
            ["CASCADE", "CONSUME"],
        )
    ),
)
def test_tx_on_PIL_images_2(
    size: t.Tuple[int],
    channels: t.Tuple[int],
    tx_mode: str,
) -> None:
    with pytest.raises(TypeError):
        if channels == (None,):
            channels = ()

        size = size + channels

        tx = ptx.ToTensor()

        img = Image.fromarray(
            np.random.randint(low=0, high=256, size=size).astype(np.uint8)
        )

        orig_params = tuple([3.0, 4.0, -1.0, 0.0])

        aug_1, params_1 = tx(img, orig_params)


@pytest.mark.parametrize(
    "size, channels, channel_first, high, high_type, tx_mode",
    list(
        product(
            [
                (
                    224,
                    224,
                ),
            ],
            [
                (None,),
                (1,),
                (3,),
                (4,),
            ],
            [True, False],
            [1.0, 256],
            [
                np.float16,
                np.float32,
                np.float64,
                np.uint8,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
            ],
            ["CASCADE", "CONSUME"],
        )
    ),
)
def test_tx_on_ndarrays(
    size: t.Tuple[int],
    channels: t.Tuple[int],
    channel_first: bool,
    high: t.Union,
    high_type: t.Any,
    tx_mode: str,
) -> None:
    if channels == (None,):
        channels = ()

    if channel_first:
        size = channels + size
    else:
        size = size + channels

    tx = ptx.ToTensor()

    if high == 1.0:
        img = np.random.uniform(low=0, high=1.0, size=size).astype(high_type)
    else:
        img = np.random.randint(low=0, high=256, size=size).astype(np.uint8)

    orig_params = tuple([3.0, 4.0, -1.0, 0.0])

    aug_1, params_1 = tx(img, orig_params)

    aug_2, params_2 = tx.cascade_transform(img, orig_params)

    aug_3, params_3 = tx.consume_transform(img, orig_params)

    assert len(list(aug_1.shape)) == 3  # [C, H, W] format ensured
    assert 0 <= torch.min(aug_1).item() <= high
    assert 0 <= torch.max(aug_1).item() <= high

    assert orig_params == params_1
    assert orig_params == params_2
    assert orig_params == params_3

    assert torch.all(torch.eq(aug_1, aug_2))
    assert torch.all(torch.eq(aug_1, aug_3))


@pytest.mark.parametrize(
    "size, channels, channel_first, high, high_type, tx_mode",
    list(
        product(
            [
                (
                    224,
                    224,
                ),
            ],
            [
                (None,),
                (1,),
                (3,),
                (4,),
            ],
            [True, False],
            [1.0, 256],
            [
                np.float16,
                np.float32,
                np.float64,
                np.uint8,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
            ],
            ["CASCADE", "CONSUME"],
        )
    ),
)
def test_tx_on_torch_tensors(
    size: t.Tuple[int],
    channels: t.Tuple[int],
    channel_first: bool,
    high: t.Union,
    high_type: t.Any,
    tx_mode: str,
) -> None:
    with pytest.raises(TypeError):

        if channels == (None,):
            channels = ()

        if channel_first:
            size = channels + size
        else:
            size = size + channels

        tx = ptx.ToTensor()

        if high == 1.0:
            img = torch.from_numpy(
                np.random.uniform(low=0, high=1.0, size=size).astype(high_type)
            )
        if high == 256:
            img = torch.from_numpy(
                np.random.randint(low=0, high=256, size=size).astype(np.uint8)
            )

        orig_params = tuple([3.0, 4.0, -1.0, 0.0])

        aug_1, params_1 = tx(img, orig_params)


# Main.
if __name__ == "__main__":
    pass

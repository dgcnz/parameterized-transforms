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
    "dtype, mode",
    list(
        product(
            [
                torch.int,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.float,
                torch.float16,
                torch.float32,
                torch.float64,
            ],
            ["CASCADE", "CONSUME"],
        )
    ),
)
def test_tx_on_PIL_images(dtype: t.Any, mode: str) -> None:

    with pytest.raises(TypeError):

        tx = ptx.ConvertImageDtype(dtype=dtype, tx_mode=mode)

        img = Image.fromarray(
            np.random.randint(low=0, high=256, size=[224, 224, 3]).astype(np.uint8)
        )

        prev_params = (1, 0.0, -2.0, 3, -4)

        aug_img, params = tx(img, prev_params)


@pytest.mark.parametrize(
    "size, channels, channel_first, high, high_type, to_dtype, tx_mode",
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
            [
                torch.int,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.float,
                torch.float16,
                torch.float32,
                torch.float64,
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
    to_dtype: t.Any,
    tx_mode: str,
) -> None:
    if channels == (None,):
        channels = ()

    if channel_first:
        size = channels + size
    else:
        size = size + channels

    tx = ptx.ConvertImageDtype(dtype=to_dtype)

    if high == 1.0:
        img = np.random.uniform(low=0, high=1.0, size=size).astype(high_type)
    else:
        img = np.random.randint(low=0, high=256, size=size).astype(np.uint8)

    orig_params = tuple([3.0, 4.0, -1.0, 0.0])

    with pytest.raises(TypeError):
        aug_1, params_1 = tx(img, orig_params)


@pytest.mark.parametrize(
    "size, channels, channel_first, high, high_type, to_dtype, tx_mode",
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
            [
                torch.int,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.float,
                torch.float16,
                torch.float32,
                torch.float64,
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
    to_dtype: t.Any,
    tx_mode: str,
) -> None:
    class DummyContext(object):
        def __enter__(self):
            pass

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    if channels == (None,):
        channels = ()

    if channel_first:
        size = channels + size
    else:
        size = size + channels

    tx = ptx.ConvertImageDtype(dtype=to_dtype)

    if high == 1.0:
        img = torch.from_numpy(
            np.random.uniform(low=0, high=1.0, size=size).astype(high_type)
        )
    else:
        img = torch.from_numpy(
            np.random.randint(low=0, high=256, size=size).astype(np.uint8)
        )

    orig_params = tuple([3.0, 4.0, -1.0, 0.0])

    # Check dtypes.
    img_dtype_name = img.dtype
    out_dtype_name = to_dtype

    if (img_dtype_name, to_dtype) in [
        (torch.float32, torch.int32),
        (torch.float32, torch.int64),
        (torch.float64, torch.int64),
    ]:
        context = pytest.raises(RuntimeError)
    else:
        context = DummyContext()

    with context:

        aug_1, params_1 = tx(img, orig_params)

        aug_2, params_2 = tx.cascade_transform(img, orig_params)

        aug_3, params_3 = tx.consume_transform(img, orig_params)

        assert len(list(aug_1.shape)) == 3 or len(list(aug_1.shape)) == 2

        assert orig_params == params_1
        assert orig_params == params_2
        assert orig_params == params_3

        assert torch.all(torch.eq(aug_1, aug_2))
        assert torch.all(torch.eq(aug_1, aug_3))

        id_params = tx.get_default_params(img=img, processed=True)
        id_aug_img, id_aug_rem_params = tx.consume_transform(img=img, params=id_params)
        if id_aug_img.dtype == img.dtype:
            assert id_aug_rem_params == ()
            assert torch.all(torch.eq(id_aug_img, img))


# Main.
if __name__ == "__main__":
    pass

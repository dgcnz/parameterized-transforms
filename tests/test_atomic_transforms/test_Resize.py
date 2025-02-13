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
import torchvision.transforms.functional as tv_fn

import parameterized_transforms.transforms as ptx

import numpy as np

import PIL.Image as Image
import PIL.ImageChops as ImageChops

from itertools import product

import typing as t


@pytest.mark.parametrize(
    "size, to_size, interpolation_mode, channels, tx_mode",
    list(
        product(
            [
                (
                    224,
                    224,
                ),
            ],
            [
                [224, 224],
                [32, 32],
                (224, 224),
                (32, 32),
                (224, 32),
                [32, 28],
                (224,),
                (32,),
                [
                    224,
                ],
                [
                    32,
                ],
                224,
                32,
            ],
            [
                tv_fn.InterpolationMode.NEAREST,
                tv_fn.InterpolationMode.BILINEAR,
                tv_fn.InterpolationMode.BICUBIC,
                tv_fn.InterpolationMode.BOX,
                tv_fn.InterpolationMode.LANCZOS,
                tv_fn.InterpolationMode.HAMMING,
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
def test_tx_on_PIL_images(
    size: t.Tuple[int],
    to_size: t.Union[t.Tuple[int], t.List[int], int],
    interpolation_mode: tv_fn.InterpolationMode,
    channels: t.Tuple[int],
    tx_mode: str,
) -> None:

    if channels == (None,):
        channels = ()

    size = size + channels

    tx = ptx.Resize(
        size=to_size,
        interpolation=interpolation_mode,
        max_size=None,
        antialias=None,
        tx_mode=tx_mode,
    )

    img = Image.fromarray(
        np.random.randint(low=0, high=256, size=size).astype(np.uint8)
    )

    orig_params = tuple([3.0, 4.0, -1.0, 0.0])

    aug_1, params_1 = tx(img, orig_params)

    aug_2, params_2 = tx.cascade_transform(img, orig_params)

    aug_3, params_3 = tx.consume_transform(img, orig_params)

    assert orig_params == params_1
    assert orig_params == params_2
    assert orig_params == params_3

    assert not ImageChops.difference(aug_1, aug_2).getbbox()
    assert not ImageChops.difference(aug_1, aug_3).getbbox()

    id_params = tx.get_default_params(img=img, processed=True)
    id_aug_img, id_aug_rem_params = tx.consume_transform(img=img, params=id_params)
    if img.size == id_aug_img.size:
        assert id_aug_rem_params == ()
        assert not ImageChops.difference(id_aug_img, img).getbbox()


@pytest.mark.parametrize(
    "size, to_size, interpolation_mode, channels, tx_mode",
    list(
        product(
            [
                (
                    224,
                    224,
                ),
            ],
            [
                [224, 224],
                [32, 32],
                (224, 224),
                (32, 32),
                (224, 32),
                [32, 28],
                (224,),
                (32,),
                [
                    224,
                ],
                [
                    32,
                ],
                224,
                32,
            ],
            [
                tv_fn.InterpolationMode.NEAREST,
                tv_fn.InterpolationMode.BILINEAR,
                tv_fn.InterpolationMode.BICUBIC,
                # # The below interpolation modes do not work well.
                # tv_fn.InterpolationMode.BOX,
                # tv_fn.InterpolationMode.LANCZOS,
                # tv_fn.InterpolationMode.HAMMING,
            ],
            [
                (3,),
                (4,),
                # (None, )  # This raises errors
            ],
            ["CASCADE", "CONSUME"],
        )
    ),
)
def test_tx_on_torch_tensors(
    size: t.Tuple[int],
    to_size: t.Union[t.Tuple[int], t.List[int], int],
    interpolation_mode: tv_fn.InterpolationMode,
    channels: t.Tuple[int],
    tx_mode: str,
) -> None:
    class DummyContext(object):
        def __enter__(self):
            pass

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    if channels == (None,):
        channels = ()

    size = channels + size

    tx = ptx.Resize(
        size=to_size,
        interpolation=interpolation_mode,
        max_size=None,
        antialias=None,
        tx_mode=tx_mode,
    )

    img = torch.from_numpy(np.random.uniform(low=0, high=1.0, size=size))

    orig_params = tuple([3.0, 4.0, -1.0, 0.0])

    if interpolation_mode in [
        tv_fn.InterpolationMode.BOX,
        tv_fn.InterpolationMode.LANCZOS,
        tv_fn.InterpolationMode.HAMMING,
    ]:
        context = pytest.raises(ValueError)
    else:
        context = DummyContext()

    with context:

        aug_1, params_1 = tx(img, orig_params)

        aug_2, params_2 = tx.cascade_transform(img, orig_params)

        aug_3, params_3 = tx.consume_transform(img, orig_params)

        assert orig_params == params_1
        assert orig_params == params_2
        assert orig_params == params_3

        assert torch.all(torch.eq(aug_1, aug_2))
        assert torch.all(torch.eq(aug_1, aug_3))

        id_params = tx.get_default_params(img=img, processed=True)
        id_aug_img, id_aug_rem_params = tx.consume_transform(img=img, params=id_params)
        if img.shape == id_aug_img.shape:
            assert id_aug_rem_params == ()
            assert torch.all(torch.eq(id_aug_img, img))


# Main.
if __name__ == "__main__":
    pass

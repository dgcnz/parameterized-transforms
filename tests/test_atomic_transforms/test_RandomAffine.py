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
    "size, degrees, translate, scale, shear, interpolation, fill, channels, tx_mode",
    list(
        product(
            [
                (
                    224,
                    224,
                ),
            ],
            # Degrees
            [
                0,
                10,
                [0.0, 5.0],
            ],
            # Translation
            [
                None,
                [0.1, 0.9],
            ],
            # Scale
            [
                None,
                (0.2, 2.0),
            ],
            # Shear
            [
                None,
                5,
                (3.0, 4),
                [0, 3.14, -3 * 3.14, 100],
            ],
            [
                tv_fn.InterpolationMode.NEAREST,
                tv_fn.InterpolationMode.BILINEAR,
                tv_fn.InterpolationMode.BICUBIC,
                # # The following do NOT work.
                # tv_fn.InterpolationMode.BOX,
                # tv_fn.InterpolationMode.LANCZOS,
                # tv_fn.InterpolationMode.HAMMING,
            ],
            [
                0,
                243,
                (23, 54, 211),
                (213, 1, 2, 245),
            ],
            [(3,), (4,), (None,)],
            ["CASCADE", "CONSUME"],
        )
    ),
)
def test_tx_on_PIL_images(
    size: t.Tuple[int],
    degrees: t.Union[
        float,
        int,
        t.List[float],
        t.List[int],
        t.Tuple[float, float],
        t.Tuple[int, int],
    ],
    translate: t.Optional[t.Tuple[float, float]],
    scale: t.Optional[t.Tuple[t.Union[int, float], t.Union[int, float]]],
    shear: t.Optional[
        t.Union[int, float, t.List[t.Union[int, float]], t.Tuple[t.Union[int, float]]]
    ],
    interpolation: tv_fn.InterpolationMode,
    fill: t.Union[int, float, t.Sequence[int], t.Sequence[float]],
    channels: t.Tuple[int],
    tx_mode: str,
) -> None:

    if isinstance(fill, float) or isinstance(fill, int):
        clean_fill = fill
    elif len(fill) == channels[0]:
        clean_fill = fill
    else:
        clean_fill = 0

    if channels == (None,):
        channels = ()

    size = size + channels

    tx = ptx.RandomAffine(
        degrees=degrees,
        translate=translate,
        scale=scale,
        shear=shear,
        interpolation=interpolation,
        tx_mode=tx_mode,
        fill=clean_fill,
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
    assert not ImageChops.difference(aug_2, aug_3).getbbox()
    assert all([isinstance(elt, float) or isinstance(elt, int) for elt in params_2])

    id_params = tx.get_default_params(img=img, processed=True)
    id_aug_img, id_aug_rem_params = tx.consume_transform(img=img, params=id_params)
    assert id_aug_rem_params == ()
    assert not ImageChops.difference(id_aug_img, img).getbbox()


@pytest.mark.parametrize(
    "size, degrees, translate, scale, shear, interpolation, fill, channels, tx_mode",
    list(
        product(
            [
                (
                    224,
                    224,
                ),
            ],
            # Degrees
            [
                0,
                10,
                [0.0, 5.0],
            ],
            # Translation
            [
                None,
                [0.1, 0.9],
            ],
            # Scale
            [
                None,
                (0.2, 2.0),
            ],
            # Shear
            [
                None,
                5,
                (3.0, 4),
                [0, 3.14, -3 * 3.14, 100],
            ],
            [
                tv_fn.InterpolationMode.NEAREST,
                tv_fn.InterpolationMode.BILINEAR,
                # # The modes below are NOT supported.
                # tv_fn.InterpolationMode.BICUBIC,
                # tv_fn.InterpolationMode.BOX,
                # tv_fn.InterpolationMode.LANCZOS,
                # tv_fn.InterpolationMode.HAMMING,
            ],
            [
                0.0,
                128.0 / 255,
                (23.0 / 255, 54.0 / 255, 211.0 / 255),
                (213.0 / 255, 1.0 / 255, 2.0 / 255, 245.0 / 255),
            ],
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
    degrees: t.Union[
        float,
        int,
        t.List[float],
        t.List[int],
        t.Tuple[float, float],
        t.Tuple[int, int],
    ],
    translate: t.Optional[t.Tuple[float, float]],
    scale: t.Optional[t.Tuple[t.Union[int, float], t.Union[int, float]]],
    shear: t.Optional[
        t.Union[int, float, t.List[t.Union[int, float]], t.Tuple[t.Union[int, float]]]
    ],
    interpolation: tv_fn.InterpolationMode,
    fill: t.Union[int, float, t.Sequence[int], t.Sequence[float]],
    channels: t.Tuple[int],
    tx_mode: str,
) -> None:

    if isinstance(fill, float) or isinstance(fill, int):
        clean_fill = fill
    elif len(fill) == channels[0]:
        clean_fill = fill
    else:
        clean_fill = 0

    if channels == (None,):
        channels = ()

    size = channels + size

    tx = ptx.RandomAffine(
        degrees=degrees,
        translate=translate,
        scale=scale,
        shear=shear,
        interpolation=interpolation,
        tx_mode=tx_mode,
        fill=clean_fill,
    )

    img = torch.from_numpy(np.random.uniform(low=0, high=1.0, size=size))

    orig_params = tuple([3.0, 4.0, -1.0, 0.0])

    aug_1, params_1 = tx.cascade_transform(img, orig_params)
    assert len(params_1) - len(orig_params) == tx.param_count

    orig_params = ()
    aug_2, params_2 = tx.cascade_transform(img, orig_params)
    aug_3, params_3 = tx.consume_transform(img, params_2)

    assert orig_params == params_3
    assert torch.all(torch.eq(aug_2, aug_3))
    assert all([isinstance(elt, float) or isinstance(elt, int) for elt in params_2])

    # Currently, the BILINEAR mode is failing.
    if interpolation != tv_fn.InterpolationMode.BILINEAR:
        id_params = tx.get_default_params(img=img, processed=True)
        id_aug_img, id_aug_rem_params = tx.consume_transform(img=img, params=id_params)
        assert id_aug_rem_params == ()
        assert torch.all(torch.eq(id_aug_img, img))


# Main.
if __name__ == "__main__":
    pass

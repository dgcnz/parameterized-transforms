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


from itertools import product

import numpy as np

import parameterized_transforms.transforms as ptx
import parameterized_transforms.core as ptc

import pytest

import PIL.Image as Image
import PIL.ImageChops as ImageChops

import torch
import torchvision.transforms.functional as tv_fn

import typing as t


@pytest.mark.parametrize(
    "image_shape, alpha, sigma, interpolation, fill, tx_mode, channels",
    list(
        product(
            [
                (
                    224,
                    224,
                ),
            ],  # image_shape
            [
                0.0,
                50.0,
                [25.0, 100.0],
            ],  # alpha
            [
                1.0,
                0.5,
                5.0,
            ],  # sigma
            [
                tv_fn.InterpolationMode.NEAREST,
                tv_fn.InterpolationMode.BILINEAR,
                tv_fn.InterpolationMode.BICUBIC,
                # # The modes below give error in applying displacements.
                # # tv_fn.InterpolationMode.BOX,
                # # tv_fn.InterpolationMode.LANCZOS,
                # # tv_fn.InterpolationMode.HAMMING,
            ],  # interpolation
            [0.0,],  # fill
            ["CASCADE", "CONSUME"],  # tx_mode
            [(3,), (4,), (None,)],  # channels
        )
    ),
)
def test_tx_on_PIL_images(
    image_shape: t.Union[int, t.List[int], t.Tuple[int, int]],
    alpha: t.Union[float, t.List[float], t.Tuple[float, float]],
    sigma: t.Union[float, t.List[float], t.Tuple[float, float]],
    interpolation: tv_fn.InterpolationMode,
    fill: t.Union[float, t.Sequence[float]],
    tx_mode: ptc.TRANSFORM_MODE_TYPE,
    channels: t.Tuple[t.Optional[int]],
):
    if channels == (None,):
        channels = ()

    size = image_shape + channels

    tx = ptx.ElasticTransform(
        image_shape=image_shape,
        alpha=alpha,
        sigma=sigma,
        interpolation=interpolation,
        fill=fill,
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
    assert not ImageChops.difference(aug_2, aug_3).getbbox()
    assert all([isinstance(elt, float) or isinstance(elt, int) for elt in params_2])

    id_params = tx.get_default_params(img=img, processed=True)
    id_aug_img, id_aug_rem_params = tx.consume_transform(img=img, params=id_params)
    assert id_aug_rem_params == ()
    assert not ImageChops.difference(id_aug_img, img).getbbox()


if __name__ == "__main__":
    pass
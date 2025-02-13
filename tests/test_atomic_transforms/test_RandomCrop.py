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
    "size, to_size, padding, padding_mode, fill, channels, tx_mode",
    list(
        product(
            [
                (
                    224,
                    224,
                ),
            ],
            [
                10,
                [
                    10,
                ],
                (10,),
                [10, 10],
                (10, 10),
                (13, 10),
                [10, 14],
            ],
            [
                0,
                1,
                2,
                [
                    0,
                ],
                [
                    1,
                ],
                [
                    2,
                ],
                (0,),
                (1,),
                (2,),
                [1, 2],
                (1, 2),
                (1, 2, 3, 4),
                [2, 4, 3, 0],
            ],
            [
                "constant",
                "edge",
                "reflect",
                "symmetric",
            ],
            [
                0,
            ],
            [(3,), (4,)],
            ["CASCADE", "CONSUME"],
        )
    ),
)
def test_tx_on_PIL_images(
    size: t.Tuple[int],
    to_size: t.Union[int, t.List[int], t.Tuple[int, int]],
    padding: t.Union[int, t.List[int], t.Tuple[int, int]],
    padding_mode: str,
    fill: t.Union[int, t.List[int], t.Tuple[int, int]],
    channels: t.Tuple[int],
    tx_mode: str,
) -> None:

    # Get size comparison.
    to_size_iterable = (
        to_size
        if (isinstance(to_size, list) or isinstance(to_size, tuple))
        else (to_size,)
    )
    max_to_size = max(to_size_iterable)

    max_from_size = max(size)

    #
    class DummyContext(object):
        def __enter__(self):
            pass

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    if max_to_size > max_from_size:
        context = pytest.raises(ValueError)
    else:
        context = DummyContext()

    with context:

        if channels == (None,):
            channels = ()

        size = size + channels

        tx = ptx.RandomCrop(
            size=to_size,
            padding=padding,
            padding_mode=padding_mode,
            pad_if_needed=True,
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

        # # This transform removes information from image.
        # # Identity params here are ill-defined in general.
        # id_params = tx.get_default_params(img=img, processed=True)
        # id_aug_img, id_aug_rem_params = tx.consume_transform(img=img, params=id_params)
        # assert id_aug_rem_params == ()
        # assert not ImageChops.difference(id_aug_img, img).getbbox()


@pytest.mark.parametrize(
    "size, to_size, padding, padding_mode, fill, channels, tx_mode",
    list(
        product(
            [
                (
                    224,
                    224,
                ),
            ],
            [
                10,
                [
                    10,
                ],
                (10,),
                [10, 10],
                (10, 10),
                (13, 10),
                [10, 14],
            ],
            [
                0,
                1,
                2,
                [
                    0,
                ],
                [
                    1,
                ],
                [
                    2,
                ],
                (0,),
                (1,),
                (2,),
                [1, 2],
                (1, 2),
                (1, 2, 3, 4),
                [2, 4, 3, 0],
            ],
            [
                "constant",
                "edge",
                "reflect",
                "symmetric",
            ],
            [
                0,
            ],
            [(3,), (4,)],
            ["CASCADE", "CONSUME"],
        )
    ),
)
def test_tx_on_torch_tensors(
    size: t.Tuple[int],
    to_size: t.Union[int, t.List[int], t.Tuple[int, int]],
    padding: t.Union[int, t.List[int], t.Tuple[int, int]],
    padding_mode: str,
    fill: t.Union[int, t.List[int], t.Tuple[int, int]],
    channels: t.Tuple[int],
    tx_mode: str,
) -> None:

    # Get size comparison.
    to_size_iterable = (
        to_size
        if (isinstance(to_size, list) or isinstance(to_size, tuple))
        else (to_size,)
    )
    max_to_size = max(to_size_iterable)

    max_from_size = max(size)

    #
    class DummyContext(object):
        def __enter__(self):
            pass

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    if max_to_size > max_from_size:
        context = pytest.raises(ValueError)
    else:
        context = DummyContext()

    with context:

        if channels == (None,):
            channels = ()

        size = channels + size

        tx = ptx.RandomCrop(
            size=to_size,
            padding=padding,
            padding_mode=padding_mode,
            pad_if_needed=True,
            fill=fill,
            tx_mode=tx_mode,
        )

        img = torch.from_numpy(np.random.uniform(low=0, high=1, size=size))

        orig_params = tuple([3.0, 4.0, -1.0, 0.0])

        aug_1, params_1 = tx.cascade_transform(img, orig_params)
        assert len(params_1) - len(orig_params) == tx.param_count

        orig_params = ()
        aug_2, params_2 = tx.cascade_transform(img, orig_params)
        aug_3, params_3 = tx.consume_transform(img, params_2)

        assert orig_params == params_3
        assert torch.all(torch.eq(aug_2, aug_3))
        assert all([isinstance(elt, float) or isinstance(elt, int) for elt in params_2])

        # # This transform removes information from image.
        # # Identity params here are ill-defined in general.
        # id_params = tx.get_default_params(img=img, processed=True)
        # id_aug_img, id_aug_rem_params = tx.consume_transform(img=img, params=id_params)
        # assert id_aug_rem_params == ()
        # assert torch.all(torch.eq(id_aug_img, img))


# Main.
if __name__ == "__main__":
    pass

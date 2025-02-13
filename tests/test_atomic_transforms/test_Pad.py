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
    "size, padding, padding_mode, fill, channels, tx_mode",
    list(
        product(
            [
                (
                    224,
                    224,
                ),
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
                253,
                (
                    1,
                    2,
                    3,
                ),
                # [1, 2, 3,],  # Lists not allowed as fill values!
            ],
            [
                (3,),
            ],
            ["CASCADE", "CONSUME"],
        )
    ),
)
def test_tx_on_PIL_images_1(
    size: t.Tuple[int],
    padding: t.Union[int, t.List[int], t.Tuple[int, int]],
    padding_mode: str,
    fill: t.Union[int, t.List[int], t.Tuple[int, int]],
    channels: t.Tuple[int],
    tx_mode: str,
) -> None:

    if channels == (None,):
        channels = ()

    size = size + channels

    tx = ptx.Pad(padding=padding, padding_mode=padding_mode, fill=fill, tx_mode=tx_mode)

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
    assert not ImageChops.difference(aug_2, aug_3).getbbox()

    # # This transform does NOT have identity params.
    # id_params = tx.get_default_params(img=img, processed=True)
    # id_aug_img, id_aug_rem_params = tx.consume_transform(img=img, params=id_params)
    # assert id_aug_rem_params == ()
    # assert not ImageChops.difference(id_aug_img, img).getbbox()


@pytest.mark.parametrize(
    "size, padding, padding_mode, fill, channels, tx_mode",
    list(
        product(
            [
                (
                    224,
                    224,
                ),
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
                253,
                (23, 55, 255, 0),
                # [1, 2, 3,],  # Lists not allowed as fill values!
            ],
            [
                (4,),
            ],
            ["CASCADE", "CONSUME"],
        )
    ),
)
def test_tx_on_PIL_images_2(
    size: t.Tuple[int],
    padding: t.Union[int, t.List[int], t.Tuple[int, int]],
    padding_mode: str,
    fill: t.Union[int, t.List[int], t.Tuple[int, int]],
    channels: t.Tuple[int],
    tx_mode: str,
) -> None:

    if channels == (None,):
        channels = ()

    size = size + channels

    tx = ptx.Pad(padding=padding, padding_mode=padding_mode, fill=fill, tx_mode=tx_mode)

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
    assert not ImageChops.difference(aug_2, aug_3).getbbox()

    # # This transform does NOT have identity params.
    # id_params = tx.get_default_params(img=img, processed=True)
    # id_aug_img, id_aug_rem_params = tx.consume_transform(img=img, params=id_params)
    # assert id_aug_rem_params == ()
    # assert not ImageChops.difference(id_aug_img, img).getbbox()


@pytest.mark.parametrize(
    "size, padding, padding_mode, fill, channels, tx_mode",
    list(
        product(
            [
                (
                    224,
                    224,
                ),
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
                0.0,
                # # Other modes lead to partial errors.
                # 253./255,
                # (23./255, 55./255, 0.),
                # [.1, .2, .3,],  # Lists not allowed as fill values!
            ],
            [
                (3,),
            ],
            ["CASCADE", "CONSUME"],
        )
    ),
)
def test_tx_on_torch_tensors_1(
    size: t.Tuple[int],
    padding: t.Union[int, t.List[int], t.Tuple[int, int]],
    padding_mode: str,
    fill: t.Union[int, t.List[int], t.Tuple[int, int]],
    channels: t.Tuple[int],
    tx_mode: str,
) -> None:

    if channels == (None,):
        channels = ()

    size = channels + size

    tx = ptx.Pad(padding=padding, padding_mode=padding_mode, fill=fill, tx_mode=tx_mode)

    img = torch.from_numpy(np.random.uniform(low=0, high=1.0, size=size))

    orig_params = tuple([3.0, 4.0, -1.0, 0.0])

    aug_1, params_1 = tx(img, orig_params)

    aug_2, params_2 = tx.cascade_transform(img, orig_params)

    aug_3, params_3 = tx.consume_transform(img, orig_params)

    assert orig_params == params_1
    assert orig_params == params_2
    assert orig_params == params_3

    assert torch.all(torch.eq(aug_1, aug_2))
    assert torch.all(torch.eq(aug_2, aug_3))

    # # This transform does NOT have identity params.
    # id_params = tx.get_default_params(img=img, processed=True)
    # id_aug_img, id_aug_rem_params = tx.consume_transform(img=img, params=id_params)
    # assert id_aug_rem_params == ()
    # assert torch.all(torch.eq(id_aug_img, img))


@pytest.mark.parametrize(
    "size, padding, padding_mode, fill, channels, tx_mode",
    list(
        product(
            [
                (
                    224,
                    224,
                ),
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
                0.0,
                # # Other modes lead to partial errors.
                # 253./255,
                # (23./255, 55./255, 255./255, 0.),
                # [.1, .2, .3, .4],  # Lists not allowed as fill values!
            ],
            [
                (4,),
            ],
            ["CASCADE", "CONSUME"],
        )
    ),
)
def test_tx_on_torch_tensors_2(
    size: t.Tuple[int],
    padding: t.Union[int, t.List[int], t.Tuple[int, int]],
    padding_mode: str,
    fill: t.Union[int, t.List[int], t.Tuple[int, int]],
    channels: t.Tuple[int],
    tx_mode: str,
) -> None:

    if channels == (None,):
        channels = ()

    size = channels + size

    tx = ptx.Pad(padding=padding, padding_mode=padding_mode, fill=fill, tx_mode=tx_mode)

    img = torch.from_numpy(np.random.uniform(low=0, high=1.0, size=size))

    orig_params = tuple([3.0, 4.0, -1.0, 0.0])

    aug_1, params_1 = tx(img, orig_params)

    aug_2, params_2 = tx.cascade_transform(img, orig_params)

    aug_3, params_3 = tx.consume_transform(img, orig_params)

    assert orig_params == params_1
    assert orig_params == params_2
    assert orig_params == params_3

    assert torch.all(torch.eq(aug_1, aug_2))
    assert torch.all(torch.eq(aug_2, aug_3))

    # # This transform does NOT have identity params.
    # id_params = tx.get_default_params(img=img, processed=True)
    # id_aug_img, id_aug_rem_params = tx.consume_transform(img=img, params=id_params)
    # assert id_aug_rem_params == ()
    # assert torch.all(torch.eq(id_aug_img, img))


# Main.
if __name__ == "__main__":
    pass

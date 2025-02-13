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
    "size, repeat_count, core_txs, channels, tx_mode",
    list(
        product(
            [
                (
                    224,
                    224,
                ),
            ],
            # repeat count to try out a large number of orderings
            [idx for idx in range(5)],
            [
                [
                    ptx.RandomHorizontalFlip(p=0.5),
                    ptx.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                    ),
                    ptx.RandomGrayscale(p=0.5),
                ],
                [
                    ptx.RandomHorizontalFlip(p=0.5),
                    ptx.ColorJitter(
                        brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2
                    ),
                    ptx.RandomGrayscale(p=0.5),
                ],
            ],
            [
                (3,),
                # (4,),
                (None,),
            ],
            ["CASCADE", "CONSUME"],
        )
    ),
)
def test_tx_on_PIL_images(
    size: t.Tuple[int],
    repeat_count: int,
    core_txs: t.List[t.Callable],
    channels: t.Tuple[int],
    tx_mode: str,
) -> None:

    if channels == (None,):
        channels = ()

    size = size + channels

    tx = ptx.RandomOrder(transforms=core_txs, tx_mode=tx_mode)

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
    if isinstance(aug_2, torch.Tensor):
        assert torch.all(torch.eq(aug_2, aug_3))
    else:
        assert not ImageChops.difference(aug_2, aug_3).getbbox()
    assert all([isinstance(elt, float) or isinstance(elt, int) for elt in params_2])

    id_params = tx.get_default_params(img=img, processed=True)
    id_aug_img, id_aug_rem_params = tx.consume_transform(img=img, params=id_params)
    assert id_aug_rem_params == ()
    assert not ImageChops.difference(id_aug_img, img).getbbox()


@pytest.mark.parametrize(
    "size, repeat_count, core_txs, channels, tx_mode",
    list(
        product(
            [
                (
                    224,
                    224,
                ),
            ],
            # repeat count to try out a large number of orderings
            [idx for idx in range(5)],
            [
                [
                    ptx.RandomHorizontalFlip(p=0.5),
                    ptx.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                    ),
                    ptx.RandomGrayscale(p=0.5),
                ],
                [
                    ptx.RandomHorizontalFlip(p=0.5),
                    ptx.ColorJitter(
                        brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2
                    ),
                    ptx.RandomGrayscale(p=0.5),
                ],
            ],
            [
                (3,),
            ],
            ["CASCADE", "CONSUME"],
        )
    ),
)
def test_tx_on_torch_tensors(
    size: t.Tuple[int],
    repeat_count: int,
    core_txs: t.List[t.Callable],
    channels: t.Tuple[int],
    tx_mode: str,
) -> None:

    if channels == (None,):
        channels = ()

    size = channels + size

    tx = ptx.RandomOrder(transforms=core_txs, tx_mode=tx_mode)

    img = torch.from_numpy(
        np.random.uniform(low=0.0, high=1.0, size=size).astype(np.uint8)
    )

    orig_params = tuple([3.0, 4.0, -1.0, 0.0])

    aug_1, params_1 = tx.cascade_transform(img, orig_params)
    assert len(params_1) - len(orig_params) == tx.param_count

    orig_params = ()
    aug_2, params_2 = tx.cascade_transform(img, orig_params)
    aug_3, params_3 = tx.consume_transform(img, params_2)

    assert orig_params == params_3
    if isinstance(aug_2, torch.Tensor):
        assert torch.all(torch.eq(aug_2, aug_3))
    else:
        assert not ImageChops.difference(aug_2, aug_3).getbbox()
    assert all([isinstance(elt, float) or isinstance(elt, int) for elt in params_2])

    id_params = tx.get_default_params(img=img, processed=True)
    id_aug_img, id_aug_rem_params = tx.consume_transform(img=img, params=id_params)
    assert id_aug_rem_params == ()
    assert torch.all(torch.eq(id_aug_img, img))


# Main.
if __name__ == "__main__":
    pass

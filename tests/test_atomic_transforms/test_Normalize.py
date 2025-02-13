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
    "size, channels, mean, std, tx_mode",
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
            [
                torch.Tensor(
                    3,
                ).uniform_(),
                np.random.uniform(size=(3,)),
                list(np.random.uniform(size=(3,))),
                tuple(np.random.uniform(size=(3,))),
            ],
            [
                torch.Tensor(
                    3,
                ).uniform_(),
                np.random.uniform(size=(3,)),
                list(np.random.uniform(size=(3,))),
                tuple(np.random.uniform(size=(3,))),
            ],
            ["CASCADE", "CONSUME"],
        )
    ),
)
def test_tx_on_PIL_images(
    size: t.Tuple[int],
    channels: t.Tuple[int],
    mean: t.Union[torch.Tensor, np.ndarray, t.List, t.Tuple],
    std: t.Union[torch.Tensor, np.ndarray, t.List, t.Tuple],
    tx_mode: str,
) -> None:

    if channels == (None,):
        channels = ()

    size = size + channels

    tx = ptx.Normalize(mean=mean, std=std, inplace=False, tx_mode=tx_mode)

    img = Image.fromarray(
        np.random.randint(low=0, high=256, size=size).astype(np.uint8)
    )

    orig_params = tuple([3.0, 4.0, -1.0, 0.0])

    with pytest.raises(TypeError):

        aug_1, params_1 = tx(img, orig_params)


@pytest.mark.parametrize(
    "size, channels, mean, std, tx_mode",
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
            [
                torch.Tensor(
                    3,
                ).uniform_(),
                np.random.uniform(size=(3,)),
                list(np.random.uniform(size=(3,))),
                tuple(np.random.uniform(size=(3,))),
            ],
            [
                torch.Tensor(
                    3,
                ).uniform_(),
                np.random.uniform(size=(3,)),
                list(np.random.uniform(size=(3,))),
                tuple(np.random.uniform(size=(3,))),
            ],
            ["CASCADE", "CONSUME"],
        )
    ),
)
def test_tx_on_ndarrays(
    size: t.Tuple[int],
    channels: t.Tuple[int],
    mean: t.Union[torch.Tensor, np.ndarray, t.List, t.Tuple],
    std: t.Union[torch.Tensor, np.ndarray, t.List, t.Tuple],
    tx_mode: str,
) -> None:

    if channels == (None,):
        channels = ()

    size = size + channels

    tx = ptx.Normalize(mean=mean, std=std, inplace=False, tx_mode=tx_mode)

    img = np.random.randint(low=0, high=256, size=size).astype(np.uint8)

    orig_params = tuple([3.0, 4.0, -1.0, 0.0])

    with pytest.raises(TypeError):

        aug_1, params_1 = tx(img, orig_params)


@pytest.mark.parametrize(
    "size, channels, mean, std, tx_mode, from_dtype",
    list(
        product(
            [
                (
                    224,
                    224,
                ),
            ],
            [
                (3,),
            ],
            [
                torch.Tensor(
                    3,
                ).uniform_(),
                np.random.uniform(size=(3,)),
                list(np.random.uniform(size=(3,))),
                tuple(np.random.uniform(size=(3,))),
            ],
            [
                torch.Tensor(
                    3,
                ).uniform_(),
                np.random.uniform(size=(3,)),
                list(np.random.uniform(size=(3,))),
                tuple(np.random.uniform(size=(3,))),
            ],
            ["CASCADE", "CONSUME"],
            [np.float16, np.float32, np.float64],
        )
    ),
)
def test_tx_on_torch_tensors_1(
    size: t.Tuple[int],
    channels: t.Tuple[int],
    mean: t.Union[torch.Tensor, np.ndarray, t.List, t.Tuple],
    std: t.Union[torch.Tensor, np.ndarray, t.List, t.Tuple],
    tx_mode: str,
    from_dtype: t.Any,
) -> None:

    if channels == (None,):
        channels = ()

    size = channels + size

    tx = ptx.Normalize(mean=mean, std=std, inplace=False, tx_mode=tx_mode)

    # high=256 fails
    img = torch.from_numpy(
        np.random.randint(low=0, high=1.0, size=size).astype(from_dtype)
    )

    orig_params = tuple([3.0, 4.0, -1.0, 0.0])

    aug_1, params_1 = tx(img, orig_params)

    aug_2, params_2 = tx.cascade_transform(img, orig_params)

    aug_3, params_3 = tx.consume_transform(img, orig_params)

    assert len(list(aug_1.shape)) == 3  # [C, H, W] format ensured
    assert aug_1.shape[0] == 3

    assert orig_params == params_1
    assert orig_params == params_2
    assert orig_params == params_3

    assert torch.all(torch.eq(aug_1, aug_2))
    assert torch.all(torch.eq(aug_1, aug_3))


@pytest.mark.parametrize(
    "size, channels, mean, std, tx_mode, from_dtype",
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
                # (None, )  # This gives error.
            ],
            [
                torch.Tensor(
                    1,
                ).uniform_(),
                np.random.uniform(size=(1,)),
                list(np.random.uniform(size=(1,))),
                tuple(np.random.uniform(size=(1,))),
            ],
            [
                torch.Tensor(
                    1,
                ).uniform_(),
                np.random.uniform(size=(1,)),
                list(np.random.uniform(size=(1,))),
                tuple(np.random.uniform(size=(1,))),
            ],
            ["CASCADE", "CONSUME"],
            [np.float16, np.float32, np.float64],
        )
    ),
)
def test_tx_on_torch_tensors_2(
    size: t.Tuple[int],
    channels: t.Tuple[int],
    mean: t.Union[torch.Tensor, np.ndarray, t.List, t.Tuple],
    std: t.Union[torch.Tensor, np.ndarray, t.List, t.Tuple],
    tx_mode: str,
    from_dtype: t.Any,
) -> None:

    if channels == (None,):
        channels = ()

    size = channels + size

    tx = ptx.Normalize(mean=mean, std=std, inplace=False, tx_mode=tx_mode)

    # high=256 fails
    img = torch.from_numpy(
        np.random.randint(low=0, high=1.0, size=size).astype(from_dtype)
    )

    orig_params = tuple([3.0, 4.0, -1.0, 0.0])

    aug_1, params_1 = tx(img, orig_params)

    aug_2, params_2 = tx.cascade_transform(img, orig_params)

    aug_3, params_3 = tx.consume_transform(img, orig_params)

    assert len(list(aug_1.shape)) == 3  # [C, H, W] format ensured
    assert aug_1.shape[0] == 1

    assert orig_params == params_1
    assert orig_params == params_2
    assert orig_params == params_3

    assert torch.all(torch.eq(aug_1, aug_2))
    assert torch.all(torch.eq(aug_1, aug_3))

    # # This transform changes images.
    # id_params = tx.get_default_params(img=img, processed=True)
    # id_aug_img, id_aug_rem_params = tx.consume_transform(img=img, params=id_params)
    # assert id_aug_rem_params == ()
    # assert torch.all(torch.eq(id_aug_img, img))


# Main.
if __name__ == "__main__":
    pass

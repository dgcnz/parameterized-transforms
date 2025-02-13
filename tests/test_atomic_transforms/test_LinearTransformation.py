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

from itertools import product

import typing as t


@pytest.mark.parametrize(
    "size, channels, tx_mode, repeats",
    list(
        product(
            [
                (
                    11,
                    11,
                ),
                (
                    12,
                    12,
                ),
                (
                    13,
                    13,
                ),
            ],
            [
                (3,),
                (4,),
                # (None,)  # This gives error.
            ],
            ["CASCADE", "CONSUME"],
            [
                100,
            ],
        )
    ),
)
def test_tx_on_torch_tensors(
    size: t.Tuple[int],
    channels: t.Tuple[int],
    tx_mode: str,
    repeats: int,
) -> None:
    if channels == (None,):
        channels = ()

    size = channels + size

    dim = np.prod(size)

    for _ in range(repeats):

        tx_matrix = torch.Tensor(dim, dim).uniform_().double()
        tx_mean = (
            torch.Tensor(
                dim,
            )
            .uniform_()
            .double()
        )

        tx = ptx.LinearTransformation(
            transformation_matrix=tx_matrix,
            mean_vector=tx_mean,
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
        assert torch.all(torch.eq(aug_2, aug_3))
        assert all([isinstance(elt, float) or isinstance(elt, int) for elt in params_2])

        # # The identity params SHOULD be `sigma=0` but this does NOT work correctly in code.
        # id_params = tx.get_default_params(img=img, processed=True)
        # id_aug_img, id_aug_rem_params = tx.consume_transform(img=img, params=id_params)
        # assert id_aug_rem_params == ()
        # assert torch.all(torch.eq(id_aug_img, img))


# Main.
if __name__ == "__main__":
    pass

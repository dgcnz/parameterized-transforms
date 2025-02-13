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
    "repetition, size, distortion_scale, p, interpolation_mode, fill, channels, tx_mode",
    list(
        product(
            [idx for idx in range(2)],
            [
                (
                    224,
                    224,
                ),
            ],
            [0.0, 0.5, 1.0],
            [0.0, 0.5, 1.0],
            [
                tv_fn.InterpolationMode.NEAREST,
                tv_fn.InterpolationMode.BILINEAR,
                tv_fn.InterpolationMode.BICUBIC,
                # # The modes below do NOT work.
                # tv_fn.InterpolationMode.BOX,
                # tv_fn.InterpolationMode.HAMMING,
                # tv_fn.InterpolationMode.LANCZOS,
            ],
            [
                0,
                243,
                (23, 54, 211),
                (213, 1, 2, 245),
            ],
            [
                # (3,),  # This channel does NOT work
                # (4,),  # This channel does NOT work
                (None,),  # Random non-reproducible error observed
            ],
            ["CASCADE", "CONSUME"],
        )
    ),
)
def test_tx_on_PIL_images(
    repetition: int,
    size: t.Tuple[int],
    distortion_scale: float,
    p: float,
    interpolation_mode: tv_fn.InterpolationMode,
    fill: t.Union[int, float, t.Sequence[int], t.Sequence[float]],
    channels: t.Tuple[int],
    tx_mode: str,
) -> None:

    try:

        if isinstance(fill, float) or isinstance(fill, int):
            clean_fill = fill
        elif len(fill) == channels[0]:
            clean_fill = fill
        else:
            clean_fill = 0

        if channels == (None,):
            channels = ()

        size = size + channels

        tx = ptx.RandomPerspective(
            distortion_scale=distortion_scale,
            interpolation=interpolation_mode,
            fill=clean_fill,
            p=p,
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

    except torch._C._LinAlgError as e:  # noqa
        raise RuntimeError(f"ERROR | `torch._C._LinAlgError` raised in the test: {e}")

    except Exception as e:
        raise RuntimeError(f"ERROR | Issue raised in the test: {e}")


@pytest.mark.parametrize(
    "repetition, size, distortion_scale, p, interpolation_mode, fill, channels, tx_mode",
    list(
        product(
            [idx for idx in range(2)],
            [
                (
                    224,
                    224,
                ),
            ],
            [0.0, 0.5, 1.0],
            [0.0, 0.5, 1.0],
            [
                tv_fn.InterpolationMode.NEAREST,
                tv_fn.InterpolationMode.BILINEAR,
                # # The modes below are NOT supported.
                # tv_fn.InterpolationMode.BICUBIC,
                # tv_fn.InterpolationMode.BOX,
                # tv_fn.InterpolationMode.HAMMING,
                # tv_fn.InterpolationMode.LANCZOS,
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
                # (None,),  # Leads to errors in general. Avoiding.
            ],
            ["CASCADE", "CONSUME"],
        )
    ),
)
def test_tx_on_torch_tensors(
    repetition: int,
    size: t.Tuple[int],
    distortion_scale: float,
    p: float,
    interpolation_mode: tv_fn.InterpolationMode,
    fill: t.Union[int, float, t.Sequence[int], t.Sequence[float]],
    channels: t.Tuple[int],
    tx_mode: str,
) -> None:
    """
    Observed issue--
    ```
    torch._C._LinAlgError: torch.linalg.lstsq:
    The least squares solution could not be computed because
    the input matrix does not have full rank (error code: 8).
    ```
    Currently, the work-around is to ignore this error if it pops up.
    """

    try:

        if isinstance(fill, float) or isinstance(fill, int):
            clean_fill = fill
        elif len(fill) == channels[0]:
            clean_fill = fill
        else:
            clean_fill = 0

        if channels == (None,):
            channels = ()

        size = channels + size

        tx = ptx.RandomPerspective(
            distortion_scale=distortion_scale,
            interpolation=interpolation_mode,
            fill=clean_fill,
            p=p,
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

        if interpolation_mode != tv_fn.InterpolationMode.BILINEAR:
            torch.all(torch.isclose(aug_2, aug_3, rtol=1e-02, atol=1e-05))
            assert all(
                [isinstance(elt, float) or isinstance(elt, int) for elt in params_2]
            )

            id_params = tx.get_default_params(img=img, processed=True)
            id_aug_img, id_aug_rem_params = tx.consume_transform(
                img=img, params=id_params
            )
            assert id_aug_rem_params == ()
            assert torch.all(torch.isclose(id_aug_img, img, rtol=1e-02, atol=1e-05))

    except torch._C._LinAlgError as e:  # noqa
        raise RuntimeError(f"ERROR | torch._C._LinAlgError raised in the test: {e}")

    except Exception as e:
        raise RuntimeError(f"ERROR | Issue raised in the test: {e}")


# Main.
if __name__ == "__main__":
    pass

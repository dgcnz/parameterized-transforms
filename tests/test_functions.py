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


import parameterized_transforms.core as ptc
import parameterized_transforms.utils as ptu

import numpy as np


def test_get_total_params_count() -> None:
    class Object:
        def __init__(self, param_count: int) -> None:
            self.param_count = param_count

        def __str__(self) -> str:
            return self.__repr__()

        def __repr__(self) -> str:
            return f"{self.__class__.__name__}" f"(param_count={self.param_count})"

    get_total_params_count = ptu.get_total_params_count

    for NUM_OBJ in [10, 100]:

        obj_list = [
            Object(param_count=np.random.randint(0, 10)) for _ in range(NUM_OBJ)
        ]
        # print(obj_list)

        obj_tuple = tuple(
            Object(param_count=np.random.randint(0, 10)) for _ in range(NUM_OBJ)
        )
        # print(obj_tuple)

        obj_dict = dict(
            [("", Object(param_count=np.random.randint(0, 10))) for _ in range(NUM_OBJ)]
        )
        # print(obj_dict)

        assert get_total_params_count(obj_list) == sum(
            [obj.param_count for obj in obj_list]
        )
        assert get_total_params_count(obj_tuple) == sum(
            [obj.param_count for obj in obj_tuple]
        )
        assert get_total_params_count(obj_dict) == sum(
            [obj.param_count for obj_name, obj in obj_dict.items()]
        )


def test_concat_params() -> None:

    concat_params = ptc.Transform.concat_params

    obj_tuple = [
        (),
        (
            1,
            2,
            3,
        ),
        (4,),
        (),
        (),
        (5, 6),
        (
            7,
            8,
        ),
    ]
    assert concat_params(*obj_tuple) == (1, 2, 3, 4, 5, 6, 7, 8)

    params_1, params_2, params_3, params_4 = (
        (),
        (1, 2),
        (
            3,
            4,
            5,
        ),
        (),
    )
    assert concat_params(params_1, params_2, params_3, params_4) == (1, 2, 3, 4, 5)


# Main.
if __name__ == "__main__":
    pass

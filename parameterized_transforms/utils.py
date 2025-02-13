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


import enum
import typing as t

from parameterized_transforms.core import SCALAR_TYPE

SCALAR_RANGE_TYPE = t.Union[t.List[SCALAR_TYPE], t.Tuple[SCALAR_TYPE, SCALAR_TYPE]]


# --------------------------------------------------------------------------------
# Transforms-related functionality.
# --------------------------------------------------------------------------------


def get_total_params_count(transforms: t.Any) -> int:
    """Returns the total number of processed parameters for the given
    collection of core transformations.

    :param transforms: The collection of core transforms.

    :return: The total number of processed parameters.
    """
    param_count: int = 0

    if isinstance(transforms, list) or isinstance(transforms, tuple):

        for transform in transforms:

            try:
                transform_param_count = transform.param_count
            except Exception as e:
                raise AttributeError(
                    "ERROR | `param_count` access "
                    f"for transform: {transform} failed; "
                    f"hit error:\n{e}"
                )

            param_count += transform_param_count

    elif isinstance(transforms, dict):

        for _, transform in transforms.items():

            try:
                transform_param_count = transform.param_count
            except Exception as e:
                raise AttributeError(
                    f"ERROR | `param_count` access "
                    f"for transforms: {transform} failed; "
                    f"hit error:\n{e}"
                )

            param_count += transform_param_count

    else:

        try:
            param_count += transforms.param_count
        except Exception as e:
            raise AttributeError(
                "ERROR | `param_count` access"
                f"for transforms: {transforms} failed; "
                f"hit error:\n{e}"
            )

    return param_count


# --------------------------------------------------------------------------------
# String representation manipulations.
# --------------------------------------------------------------------------------


def indent(data: str, indentor: str = "  ", connector: str = "\n") -> str:
    """Indents the given string data with given `indentor`.

    :param data: The data to indent.
    :param indentor: The indenting character.
        DEFAULT: `"  "` (two spaces).
    :param connector: The string used to concatenate given components.
        DEFAULT: `"\n"`.

    :return: The indented data.

    Careful not to have new-line characters in the `indentor`. This is
    currently allowed but NOT tested, use at your own risk.
    """
    lines = string_to_components(string=data, separator=connector)
    indented_lines = indent_lines(lines=lines, indentor=indentor)
    indented_data = components_to_string(components=indented_lines, connector=connector)

    return indented_data


def indent_lines(lines: t.List[str], indentor: str = "  ") -> t.List[str]:
    """Adds as prefix the `indentor` string to each of the given lines.

    :param lines: The lines of the content.
    :param indentor: The indenting character.
            DEFAULT: `"  "` (two spaces).

    :return: The indented lines.

    Careful not to have new-line characters in the `indentor`. This is
    currently allowed but NOT tested, use at your own risk.
    """

    return [indent_line(line=line, indentor=indentor) for line in lines]


def indent_line(line: str, indentor: str = "  ") -> str:
    """Indents the given `string` using the given `indentor`.

    :param line: The string to be indented.
    :param indentor: The indenting string.

    :returns: The indented string.
    """

    return f"{indentor}{line}"


def string_to_components(string: str, separator: str = "\n") -> t.List[str]:
    """Split a given string into componenets based on given `separator`.

    :param string: The input string.
    :param separator: The string to split the given string into components.

    :returns: The components extracted from this string.
    """

    return string.split(separator)


def components_to_string(components: t.List[str], connector: str = ",\n") -> str:
    """Concatenate given components into a string using given `connector`.

    :param components: The components of a string.
    :param connector: The string used to concatenate given components.

    :returns: The concatenated string made from the given `components` using
    the given `connector` string.
    """
    return connector.join(components)

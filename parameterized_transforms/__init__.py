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


from parameterized_transforms import core
from parameterized_transforms import transforms
from parameterized_transforms import utils


ATOMIC_TRANSFORMS = {
    "ToTensor": transforms.ToTensor,
    "PILToTensor": transforms.PILToTensor,
    "ConvertImageDtype": transforms.ConvertImageDtype,
    "ToPILImage": transforms.ToPILImage,
    "Normalize": transforms.Normalize,
    "Resize": transforms.Resize,
    "CenterCrop": transforms.CenterCrop,
    "Pad": transforms.Pad,
    "Lambda": transforms.Lambda,
    "RandomCrop": transforms.RandomCrop,
    "RandomHorizontalFlip": transforms.RandomHorizontalFlip,
    "RandomVerticalFlip": transforms.RandomVerticalFlip,
    "RandomPerspective": transforms.RandomPerspective,
    "RandomResizedCrop": transforms.RandomResizedCrop,
    "FiveCrop": transforms.FiveCrop,
    "TenCrop": transforms.TenCrop,
    "LinearTransformation": transforms.LinearTransformation,
    "ColorJitter": transforms.ColorJitter,
    "RandomRotation": transforms.RandomRotation,
    "RandomAffine": transforms.RandomAffine,
    "Grayscale": transforms.Grayscale,
    "RandomGrayscale": transforms.RandomGrayscale,
    "RandomErasing": transforms.RandomErasing,
    "GaussianBlur": transforms.GaussianBlur,
    "RandomInvert": transforms.RandomInvert,
    "RandomPosterize": transforms.RandomPosterize,
    "RandomSolarize": transforms.RandomSolarize,
    "RandomAdjustSharpness": transforms.RandomAdjustSharpness,
    "RandomAutocontrast": transforms.RandomAutocontrast,
    "RandomEqualize": transforms.RandomEqualize,
    "ElasticTransform": transforms.ElasticTransform,
}


COMPOSING_TRANSFORMS = {
    "Compose": transforms.Compose,
    "RandomApply": transforms.RandomApply,
    "RandomOrder": transforms.RandomOrder,
    "RandomChoice": transforms.RandomChoice,
}


TRANSFORMS = dict(
    tuple(ATOMIC_TRANSFORMS.items())
    + tuple(COMPOSING_TRANSFORMS.items())
)

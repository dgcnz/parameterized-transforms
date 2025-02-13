# A Brief Introduction to the Transforms in this Package

- 5 Minute Read


<a name="summary"></a>
## Summary 
* In this tutorial, we will briefly demonstrate some of the parameterized transforms that we provide in this package.
* For each `torchvision`-based transform, we provide one parameterized version with the same name that exposes the parameters of this transform.
* In addition, we provide useful tools and wrappers in order to ease working with parameterized transforms and porting to-and-from `torch`/`torchvision`-based code. 


<a name="torchvision-transforms-transforms"></a>
## Transforms from `torchvision` but Parametrized!
* **Please refer to the files
[transforms.py](https://github.com/apple/parameterized-transforms/blob/main/parameterized_transforms/transforms.py) for more information while reading through this section.**


* Note that `torchvision.transforms.transform` provides `35` transforms. 
Out of these, `31` are atomic transforms and the remaining `4` are composing transforms.
All these transforms are listed below.


**Atomic transforms from `torchvision/transforms/transforms.py` (`31` transforms)**

|                  |                        |                         |                        |                             |
|------------------|------------------------|-------------------------|------------------------|-----------------------------|
| `CenterCrop`     | `ColorJitter`          | `ConvertImageDtype`     | `ElasticTransform`     | `FiveCrop`                  |
| `GaussianBlur`   | `Grayscale`            | `Lambda`                | `LinearTransformation` | `Normalize`                 |      
| `PILToTensor`    | `Pad`                  | `RandomAdjustSharpness` | `RandomAffine`         | `RandomAutocontrast`        | 
| `RandomCrop`     | `RandomEqualize`       | `RandomErasing`         | `RandomGrayscale`      | `RandomHorizontalFlip`      | 
| `RandomInvert`   | `RandomPerspective`    | `RandomPosterize`       | `RandomResizedCrop`    | `RandomRotation`            | 
| `RandomSolarize` | `RandomVerticalFlip`   | `Resize`                | `TenCrop`              | `ToPILImage`                | 
| `ToTensor`       |                        |                         |                        |                             |


**Composing transforms from `torchvision/transforms/transforms.py` (`4` transforms)**

|           |               |                |               |
|-----------|---------------|----------------|---------------|
| `Compose` | `RandomApply` | `RandomChoice` | `RandomOrder` |


* For each transform provided in `torchvision/transforms/transforms.py`, we provide its parameterized counterpart in 
`parameterized_transforms/transforms.py`.
These parameterized transforms behave exactly identical to their `torchvision` counterparts in terms of the augmentation 
functionality but provide access to their augmentation parameters as well.
All these transforms that are provided in `parameterized_transforms/transforms.py` are listed below--


**Atomic transforms from [parameterized_transforms/transforms.py](https://github.com/apple/parameterized-transforms/blob/main/parameterized_transforms/transforms.py) (`31` transforms)**

|                  |                        |                         |                        |                             |
|------------------|------------------------|-------------------------|------------------------|-----------------------------|
| `CenterCrop`     | `ColorJitter`          | `ConvertImageDtype`     | `ElasticTransform`     | `FiveCrop`                  |
| `GaussianBlur`   | `Grayscale`            | `Lambda`                | `LinearTransformation` | `Normalize`                 |      
| `PILToTensor`    | `Pad`                  | `RandomAdjustSharpness` | `RandomAffine`         | `RandomAutocontrast`        | 
| `RandomCrop`     | `RandomEqualize`       | `RandomErasing`         | `RandomGrayscale`      | `RandomHorizontalFlip`      | 
| `RandomInvert`   | `RandomPerspective`    | `RandomPosterize`       | `RandomResizedCrop`    | `RandomRotation`            | 
| `RandomSolarize` | `RandomVerticalFlip`   | `Resize`                | `TenCrop`              | `ToPILImage`                | 
| `ToTensor`       |                        |                         |                        |                             |


**Composing transforms from [parameterized_transforms/transforms.py](https://github.com/apple/parameterized-transforms/blob/main/parameterized_transforms/transforms.py) (`4` transforms)**

|           |               |                |               |
|-----------|---------------|----------------|---------------|
| `Compose` | `RandomApply` | `RandomChoice` | `RandomOrder` |



<a name="useful-wrapper-transforms"></a>
## Useful Wrapper Transforms

* **Please refer to the file
[wrappers.py](https://github.com/apple/parameterized-transforms/blob/main/parameterized_transforms/wrappers.py)
for more information while reading through this section.**

* In addition to these parameterized transforms, we provide extra wrapper transforms that allow for miscellaneous useful functionality.
In particular, we provide `4` useful wrapper transforms which are listed below--

|                        |                      |                      |              |
|------------------------|----------------------|----------------------|--------------|
| `ExtractDefaultParams` | `ApplyDefaultParams` | `CastParamsToTensor` | `DropParams` |


* These transforms are mostly intended to ease manipulation of train/test augmentation stacks and for porting to-and-from 
`torch`/`torchvision`-based codebases.
We briefly describe their usage below.
* `ExtractDefaultParams` wrapper transform stores an atomic or a composing transform as its core transform.
When given an image `img` and parameters `params` as input, it fetches the default parameters of the core transform, 
appends them to the input parameters to obtain concatenated parameters `concat_params`, and returns the tuple of 1. the 
input image `img` as it is, and 2. the concatenated parameters `concat_params`.
The purpose of this wrapper transform is to expose the default parameters of any transform of interest without actually 
applying that transform.
* `ApplyDefaultParams` wrapper transform stores an atomic or a composing transform as its core transform.
It behaves identical to `ExtractDefaultParams` except for a subtle difference as suggested by its name.
When given an image `img` and parameters `params` as input, it fetches the default parameters of the core transform, 
applies the core transform defined by these default parameters on the `img` to get the default-augmented image `aug_img`, 
appends the default parameters to the input parameters in order to obtain concatenated parameters `concat_params`, and 
returns the tuple of 1. the default augmented image `aug_img`, and 2. the concatenated parameters `concat_params`.
    * The reason to provide this subtly different wrapper transform is that there are some transforms where default 
    parameters can preserve the image information but still change the image itself (for instance, `ToTensor`, 
    `ConvertImageDtype`, `Resize`, etc.) and there are others where the information in the image is indeed lost (
    for instance, `CenterCrop`, `Grayscale`, etc.).
    Given these possibilities, this wrapper is often helpful in easily generating the test transforms corresponding to 
    the given train transforms; wrapping each component transform of a train transform stack often gives the desired 
    test transform!
* `CastParamsToTensor` wrapper transform stores an atomic or a composing transform as its core transform.
This transform is intended to wrap a finalized stack of core transforms, where the output parameter tuple is converted 
to a `torch`-tensor with `dtype=torch.float32`.
In particular, when given an image `img` and parameters `params` as input, it converts the parameters to a 
`torch`-tensor `params_torch` with `dtype=torch.float32` and returns the tuple of 1. the input image `img` as it is, 
and 2. the parameters converted to `torch`-tensor `params_torch` with `dtype=torch.float32`.
  * The utility of this wrapper transform will become clear in the tutorial on migrating code from 
  `torch`/`torchvision` to use this package!
* `DropParams` wrapper transform stores an atomic or a composing transform as its core transform.
This transform is intended to help convert a parameterized transform into its `torchvision`-counterpart to help port 
code from our package back to `torch`/`torchvision`.
As the name suggests, given given an image `img` and parameters `params` as input, it just returns the image `img` as 
it is, thereby dropping parameters.
Thus, wrapping any parametrize transform with this wrapper transform, we get its `torchvision`-counterpart!

  
* In the subsequent tutorials, we will illustrate some of the atomic, composing, and 
wrapper transforms provided in this package so that readers fill comfortable working with them!


<a name="next-tutorial-preview"></a>
## About the Next Tutorial
* Till now, we have studied the core structure of parameterized transforms, we have writte our own custom parameterized 
atomic and composing transforms, and we have briefly gone through the transforms provided in this package.
* So, in the next tutorial 
[004-Parametrized-Transforms-in-Action](004-Parametrized-Transforms-in-Action.md),
we will see the parameterized transforms in action; we will see how parameterized transforms implemented in the structure 
prescribed in this package allow us to perform cool things!
* If you were wondering why do we need to define our transform with the specific templates illustrated in the previous 
tutorials, this tutorial would answer your doubts by showing how these transforms would fit with each other seamlessly 
and with other transform provided in the package.
This should also help demonstrate the power and the capabilities of the parameterized transforms!

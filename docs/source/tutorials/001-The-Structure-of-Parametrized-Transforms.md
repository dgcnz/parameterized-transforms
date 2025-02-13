# The Structure of Parametrized Transforms

- 12 Minute Read


<a name="summary"></a>
## Summary 
* This tutorial covers the details of how parameterized transforms are implemented.
* We have two different core transform classes-- `AtomicTransform` and `ComposingTransform`.
* These two classes inherit from the `Transform` base class, which provides their common functionalities.


<a name="core"></a>
## The Core Structure of a Parametrized Transform 
* **The classes described in this tutorial are implemented in [core.py](https://github.com/apple/parameterized-transforms/blob/main/parameterized_transforms/core.py).**

<a name="the-Transform-base-class"></a>
### The `Transform` Base Class 
* A **Transform** is an object that performs some processing on an input image to modify it.


* Every transform in this package has a certain number of parameters associated with it. 
  * For instance, `Grayscale` transform always converts the input `RGB` image to grayscale and thus, has `0` parameters. 
  * On the other hand, `RandomGrayscale` transform converts the input `RGB` image to grayscale with a given probability and 
  so, it has `1` parameter to specify whether the input image was indeed converted to grayscale or not.
  * As another example, the `ColorJitter` transform perturbs the brightness, contrast, saturation, and hue of the input 
  image with randomly sampled strengths and in a random order.
  Thus, this transform has `8` parameters-- four parameters to specify the strengths of the brightness, contrast, 
  saturation, and hue perturbations, and four more parameters to indicate the order in which these perturbations were applied.
  * All provided transforms store this number of parameters as an attribute `param_count`, which is inferred using the 
  `set_param_count` method. 
  In order to write your own transforms, you MUST define this attribute while initializing your class. You *may* use 
  the provided `set_param_count` method to do so but it is NOT mandatory.


* Each transform is designed to input a tuple of the following two objects-- 1. an image, and 2. a tuple of parameters.
It also outputs a tuple of the following two objects-- 1. the augmented image, and 2. the updated parameters.
This is a major change in comparison with `torchvision`-based transforms; `torchvision`-based transforms input only an 
image and output only the processed image.
  * Check the type-hints `IMAGE_TYPE`, `PARAM_TYPE`, `TRANSFORM_RETURN_TYPE`, and the signature of the `__call__` method 
  of the base class for the same. 


* This package implements all transforms in two different **modes**-- **CASCADE** and **CONSUME**.
  * Each transform MUST have a mode and is stored in the `tx_mode` attribute. Its values are defined in the 
  `TransformMode` enum.
  * In **CASCADE** mode, the transform inputs an image and a tuple of parameters. 
  It then samples local parameters, applies the transform using these parameters, and returns a tuple of the following 
  two objects-- 1. the augmented image, and 2. the previously input parameters appended with the local parameters.
  * In **CONSUME** mode, the transform inputs an image and a tuple of parameters. 
  It then extracts the required number of local parameters from the input parameters, applies the transform with these 
  parameters, and returns a tuple of the following two objects-- 1. the augmented image, and 2. the remaining 
  parameters from the input ones.
  * These two modes provide us with enough power to extract parameters of transforms and to reproduce transforms 
  defined by given (well-defined) parametrization.  
  * The `__call__` method of the base class redirects the inputs to `cascade_transform` or `consume_transform` based on 
  the mode of the transform.


* For each transform, we define a method `get_default_params`, which is intended to return parameters that preserve 
the identity of the input image whenever possible.
If this is not possible, these parameters are intended to preserve as much information in the input image as possible.
  * For example, the `RandomRotation` transform has `param_count = 1` to capture the angle of rotation of the image. 
  Thus, the default parameters are the singleton tuple `(0, )` as a `0`-degree rotation preserves the image.
  * For the case of `ToTensor` transform being applied on input images of the class `PIL.Image.Image`, we have 
  `param_count = 0` and the default parameters are the empty tuple `()`; these parameters preserve the information in the 
  image and only change the type in which the image is represented. 
  * However, the `CenterCrop` transform differs from the above two cases.
  It has `param_count = 0` and when the desired crop size does NOT match the image size, the resulting augmented image 
  will NEVER be identical to the original image. 
  Here still, the default parameters are the empty tuple `()`; they do NOT preserve the identity of the image but 
  retain as much information as possible. 
  * Another example is the `Compose` transform, whose default parameters are the concatenation of the default parameters 
  of its components.
  * Whenever possible, the default parameters are intended to act as "identity parameters"; applying the transform with 
  those parameters retains the image information. 
  In other cases, these are the parameters that retain as much information from the image as possible.
  * Note that in some cases, it may be possible to have more than one default parameters. 
  For instance, the `ColorJitter` transform can preserve the input image as it is by applying brightness, contrast, and 
  saturation perturbations of `1.0` and hue perturbation of `0.0`. 
  However, these four operations can be applied in ANY order to get the identical image back from the transform.
  To cater for such cases, the corresponding transforms have an extra attribute `default_params_mode`.
  * The values taken by `default_params_mode` are defined in the `DefaultParamsMode` enum. 
  The value `DefaultParamsMode.RANDOMIZED` is used to obtain a randomly sampled default parameter from the set of all 
  possible default parameters whenever the transform is applied. 
  On the other hand, the other value `DefaultParamsMode.UNIQUE` is used to obtain a fixed pre-defined 
  default parameter value whenever the transform is applied. 

  
* Thus, to define your own transform, you can do the following--

> 1. **Subclass from `Transform`.**
> 2. **Define the attributes `tx_mode` and `param_count`; for the latter attribute, you may choose to do so with your concrete definition of the `set_param_count` method.**
> 3. **Define your concrete implementation of `cascade_transform` method.**
> 4. **Define your concrete implementation of `consume_transform` method.**
> 5. **Define your concrete implementation of `get_default_params` method.**
> 6. **(Optional) Define your concrete implementation of `__str__` method.**


* We classify all transforms into two major types-- **Atomic** and **Composing**. 
As the names suggest, **Atomic** transforms perform one simple set of actions on a data point.
On the other hand, **Composing** transforms input one or more other transforms (either **Atomic** or **Composing**) and 
combine their functionalities to achieve the desired compositional behavior.
We refer to these one or more transforms as the **core** transforms.
  * For example, `Grayscale` would be an **Atomic** transform that converts given `RGB` images into grayscale.
  * However, `Compose` would be a **Composing** transform that has a list of other **core** transforms and applies them 
  sequentially on a given input image. 
  * Similarly, `RandomChoice` would also be a **Composing** transform that has a list of other **core** transforms and applies a 
  randomly sampled transform from this list on a given input image.  


<a name="the-AtomicTransform-base-class"></a>
### The `AtomicTransform` Base Class 
* The atomic transforms are implemented using the base class `AtomicTransform`, which subclasses the 
base class `Transform` .
The `AtomicTransform` provides a partial implementation of `cascade_transform` and `consume_transform` in order 
to enable the functions of all the atomic transforms as described below.


* Consider an atomic transform that is being called with the input image `img` and the input parameters `params`. 
It should perform the following five steps to operate in the CASCADE mode--
  1. Generate raw parameters `local_raw_params` for applying the transform.
  2. Apply the transform with these raw parameters on `img` to obtain the augmented image `aug_img`.
  3. Post-process `local_raw_params` to obtain the processed parameters `local_proc_params`.
  4. Append `local_proc_params` to the input `params` to obtain the concatenated processed parameters `concat_proc_params`.
  5. Return the tuple of the augmented image `aug_img` and `local_proc_params`.
* These steps are captured in the partial implementation of `cascade_tranform` method of `AtomicTransform`--


```python
def cascade_transform(self, img: IMAGE_TYPE, params: PARAM_TYPE) -> TRANSFORM_RETURN_TYPE:

   local_raw_params = self.get_raw_params(img=img)
   aug_img = self.apply_transform(img=img, params=local_raw_params)
   local_proc_params = self.post_process_params(img=img, params=local_raw_params)
   concat_proc_params = Transform.concat_params(params, local_proc_params)
   
   return aug_img, concat_proc_params
```


* For the transform to operate in the CONSUME mode, the transform should perform the following steps-- 
  1. From the given parameters, extract the processed parameters `local_proc_params` for the transform and the remaining processed parameters `rem_proc_params` to be passed on.
  2. Pre-process `local_proc_params` to obtain the raw local parameters `local_raw_params`.
  3. Apply the transform with these raw parameters on `img` to obtain the augmented image `aug_img`.
  4. Return the tuple of the augmented image `aug_img` and the remaning processed parameters `rem_proc_params`.
* These steps are captured in the partial implementation of `consume_tranform` method of `AtomicTransform`--


```python
def consume_transform(self, img: IMAGE_TYPE, params: PARAM_TYPE) -> TRANSFORM_RETURN_TYPE:

   local_proc_params, rem_proc_params = self.extract_params(params=params)
   local_raw_params = self.pre_process_params(img=img, params=local_proc_params)
   aug_img = self.apply_transform(img=img, params=local_raw_params)

   return aug_img, rem_proc_params
```


* Thus, to define your own atomic transform, you can do the following-- 
> 1. **Subclass from `AtomicTransform`.**
> 2. **Define the attributes `tx_mode` and `param_count`; you may choose to do so with your concrete definition of the `set_param_count` method for the latter.**
> 3. **Define your concrete implementation of `get_raw_params` method.**
> 4. **Define your concrete implementation of `apply_transform` method.**
> 5. **Define your concrete implementation of `post_process_params` method.**
> 6. **Define your concrete implementation of `extract_params` method.**
> 7. **Define your concrete implementation of `pre_process_params` method.**
> 8. **Define your concrete implementation of `get_default_params` method.**
> 9. **(Optional) Define your concrete implementation of `__str__` method.**


<a name="the-ComposingTransform-base-class"></a>
### The `ComposingTransform` Base Class 
* The composing transforms are implemented using the base class `ComposingTransform`, which subclasses the base 
class `Transform`.
The `ComposingTransform` provides a partial implementation of `cascade_transform` and `consume_transform` in 
order to enable the composing functionalities of all the composing transforms as described below.


* Consider a compsing transform that is being called with the input image `img` and the input parameters `params`.
The composing transform is designed to perform some composing functionality on top of its core transforms and thus, it 
should perform the following five steps to operate in the CASCADE mode--
  1. Generate raw parameters `local_raw_params` to guide the application of the core transform.
  2. Apply the core transforms guided by `local_raw_params` on `img` to obtain the augmented image `aug_img` along with the parameters `aug_params` generated by the core transform.
  3. Post-process `local_raw_params` and `aug_params` to obtain their processed versions-- `local_proc_params` and `aug_proc_params` respectively.
  4. Append `local_proc_params` and `aug_proc_params` to the input `params` to obtain the concatenated processed parameters `concat_proc_params`.
  5. Return the tuple of the augmented image `aug_img` and `local_proc_params`.
* These steps are captured in the partial implementation of `cascade_tranform` method of `ComposingTransform`--


```python
def cascade_transform(self, img: IMAGE_TYPE, params: PARAM_TYPE) -> TRANSFORM_RETURN_TYPE:

   local_raw_params = self.get_raw_params(img=img)
   aug_img, aug_params = self.apply_cascade_transform(img=img, params=local_raw_params)
   local_proc_params, aug_proc_params = self.post_process_params(img=img, params=local_raw_params, aug_params=aug_params)
   concat_proc_params = Transform.concat_params(params, local_proc_params, aug_proc_params)

   return aug_img, concat_proc_params
```


* For the transform to operate in the CONSUME mode, the composing transform should perform the following steps--
  1. From the given parameters, extract the processed parameters `concat_local_params` for the transform and the remaining processed parameters `rem_proc_params` to be passed on.
  2. Pre-process `concat_local_params` to obtain the raw local parameters `local_raw_params` along with the core augmentation parameters `aug_params`.
  3. Apply the core transform guided by `local_raw_params` and defined using `aug_params` on `img` to obtain the augmented image `aug_img` and the remaining augmentation parameters `rem_aug_params`.
     * Note that `rem_aug_params` MUST be empty. Otherwise, there is an error in the implementation. 
  4. Return the tuple of the augmented image `aug_img` and the remaning processed parameters `rem_proc_params`.
* These steps are captured in the partial implementation of `consume_tranform` method of `AtomicTransform`--


```python
def consume_transform(self, img: IMAGE_TYPE, params: PARAM_TYPE) -> TRANSFORM_RETURN_TYPE:

   concat_proc_params, rem_proc_params = self.extract_params(params=params)
   local_raw_params, aug_params = self.pre_process_params(img=img, params=concat_local_params)
   aug_img, rem_aug_params = self.apply_consume_transform(img=img, params=local_raw_params, aug_params=aug_params)
   assert len(rem_aug_params) == 0     

   return aug_img, rem_proc_params
```


* Thus, to define your own composing transform, you can do the following-- 
> 1. **Subclass from `ComposingTransform`.**
> 2. **Define the attributes `tx_mode` and `param_count`; you may choose to do so with your concrete definition of the `set_param_count` method for the latter.**
> 3. **Define the core transform with the attribute `transforms`.**
> 4. **Define your concrete implementation of `get_raw_params` method.**
> 5. **Define your concrete implementation of `apply_cascade_transform` method.**
> 6. **Define your concrete implementation of `post_process_params` method.**
> 7. **Define your concrete implementation of `extract_params` method.**
> 8. **Define your concrete implementation of `pre_process_params` method.**
> 9. **Define your concrete implementation of `apply_consume_transform` method.** 
> 10. **Define your concrete implementation of `get_default_params` method.**
> 11. **(Optional) Define your concrete implementation of `__str__` method.** 


<a name="next-tutorial-preview"></a>
## About the Next Tutorial
* In the next tutorial [002-How-to-Write-Your-Own-Transforms.md](002-How-to-Write-Your-Own-Transforms.md), 
we will use the structure of parameterized transforms as described in this tutorial to write our own custom transforms! 
* In particular, we will write one atomic transform named `RandomColorErasing` and one composing transform named `RandomSubsetApply` to 
better understand the structure of parameterized transforms. 
We highly recommend spending some time going through this tutorial to understand the nitty-gritties of actually writing 
a parameterized transforms.
* However, in case you are only interested in using the parameterized transforms provided by this package, you may skip the next tutorial and jump directly to the subsequent tutorial--
[003-A-Brief-Introduction-to-the-Transforms-in-This-Package](003-A-Brief-Introduction-to-the-Transforms-in-This-Package.md).

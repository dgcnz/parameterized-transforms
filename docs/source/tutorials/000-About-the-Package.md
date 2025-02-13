# About the Package

- 5 Minute Read

<a name="summary"></a>
## Summary 
* This tutorial describes the three important aspects of the `Parameterized Transforms` package: 
  1. **Why** do we need this package?, 
  2. **What** does the package provide?, and 
  3. **How** to use the package?  


* **NOTE:** We will be using the terms **Augmentation (noun) / Augment (verb)** interchangeably with 
**Transform (noun) / Transform (verb)** throughout the tutorials.

<a name="the-why-aspect"></a>
## The *Why* Aspect
* Augmentation strategies are important in computer vision research for improving the performance of deep learning approaches.
* Popular libraries like `torchvision` and `kornia` provide implementations of widely used and important transforms.
* Many recent research ideas revolve around using the information of augmentation parameters in order to learn better representations.
In this context, different popular libraries have different pros and cons:
  * For instance, most of recent deep learning approaches define their augmentation stacks in terms of the 
  `torchvision`-based transforms, experiment with them, and report the best-performing stacks. 
  However, `torchvision`-based transforms do NOT provide access to their parameters, thereby limiting the research possibilities aimed at extracting information provided by augmentation parameters to learn better data representations.  
  * On the other hand, although `kornia`-based augmentation stacks do provide access to the parameters of the augmentations, 
  reproducing results obtained with `torchvision` stacks using `kornia`-based augmentations is difficult due to the differences in their implementation.
* Ideally, we want to have transforms implementations that have the following desired properties:
  1. they can provide access to their parameters by exposing them,
  2. they allow reproducible augmentations by enabling application of the transform defined by given parameters, 
  3. they are easy to subclass and extend in order to tweak their functionality, and
  4. they have implementations that match those of the transforms used in obtaining the state-of-the-art results (mostly, `torchvision`).
* This is very difficult to achieve with any of the currently existing libraries.


<a name="the-what-aspect"></a>
## The *What* Aspect
* What this package provides is a modular, uniform, and easily extendable skeleton with a re-implementation of `torchvision`-based 
transforms that gives you access to their augmentation parameters and allows reproducible augmentations.
* In particular, these transforms can perform two extremely crucial tasks associated with exposing their parameters:
  1. Given an image, the transform can return an augmentation along with the parameters used for the augmentation.
  2. Given an image and well-defined augmentation parameters, the transform can return the corresponding augmented image.
* The uniform template for all transforms and a modular re-implementation means that you can easily subclass the 
transforms and tweak their functionalities.
* In addition, you can write your own custom transforms using the provided templates and combine them seamlessly 
with other custom or package-defined transforms for your experimentation.


<a name="the-how-aspect"></a>
## The *How* Aspect
* To start using the package, we recommend the following-- 
  1. Read through the [Prerequisites](#prerequisites) listed below and be well-acquainted with them.
  2. [Install the Package](https://github.com/apple/parameterized-transforms/blob/main/README.md#installation) as described in the link.
  3. Read through the [Tutorial Series](#tutorials-in-a-nutshell).
  4. After that, you should be ready to write and experiment with parameterized transforms!


<a name="prerequisites"></a>
## Prerequisites
Here are the prerequisites for this package-- 
* `numpy`: being comfortable with `numpy` arrays and operations,
* `PIL`: being comfortable with basic `PIL` operations and the `PIL.Image.Image` class,
* `torch`: being comfortable with `torch` tensors and operations, and
* `torchvision`: being comfortable with `torchvision` transforms and operations.


<a name="tutorials-in-a-nutshell"></a>
## A Preview of All Tutorials
* Here is an overview of the tutorials in this series and the topics they cover--

| Title                                                                                                                                           | Contents                                                                                                                                                                                           |
|-------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [0. About the Package](000-About-the-Package.md)                                                                                                | An overview of the package                                                                                                                                                                         |
| [1. The Structure of Parametrized Transforms](001-The-Structure-of-Parametrized-Transforms.md)                                                  | Explanation of the base classes `Transform`, `AtomicTransform`, and `ComposingTransform`                                                                                                           |
| [2. How to Write Your Own Transforms](002-How-to-Write-Your-Own-Transforms.md)                                                                  | A walk-through of writing custom transforms-- an atomic transform named `RandomColorErasing` and a composing transform named `RandomSubsetApply`                                                   |
| [3. A Brief Introduction to the Transforms in This Package](003-A-Brief-Introduction-to-the-Transforms-in-This-Package.md)                      | * Information about all the transforms provided in this package <br> * Demonstrations of some of the important transforms from the package                                                         |
| [4. Parametrized Transforms in Action](004-Parametrized-Transforms-in-Action.md)                                                                | * Visualization of augmentations produced by the custom transforms <br/> * Combining custom transforms with the ones from the package <br/> * Extending transforms easily to tweak their behaviors |
| [5. Migrate From `torch`/`torchvision` to `parameterized_transforms` in Three Easy Steps](005-Migrate-To-and-From-torch-in-Three-Easy-Steps.md) | Instructions to easily modify code with `torch`-based datasets/loaders and `torchvision`-based transforms to use the parameterized transforms                                                      |


<a name="credits"></a>
## Credits
In case you find our work useful in your research, you can use the following `bibtex` entry to cite us--
```text
@software{Dhekane_Parameterized_Transforms_2025,
    author = {Dhekane, Eeshan Gunesh},
    month = {2},
    title = {{Parameterized Transforms}},
    url = {https://github.com/apple/parameterized-transforms},
    version = {1.0.0},
    year = {2025}
}
```


<a name="next-tutorial-preview"></a>
## About the Next Tutorial
* In the next tutorial [001-The-Structure-of-Parametrized-Transforms.md](001-The-Structure-of-Parametrized-Transforms.md), we will describe the core structure of parameterized transforms.
* We will see two different types of transforms, **Atomic** and **Composing**, and describe their details.

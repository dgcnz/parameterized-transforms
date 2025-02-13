# Parameterized Transforms


## Index
1. [About the Package](#about-the-package)
2. [Installation](#installation)
3. [Getting Started](#getting-started)


<a name="about-the-package"></a>
## About the Package
* The package provides a uniform, modular, and easily extendable implementation of `torchvision`-based transforms that provides access to their parameterization.
* With this access, the transforms enable users to achieve the following two important functionalities--
  * Given an image, the transform can return an augmentation along with the parameters used for the augmentation.
  * Given an image and augmentation parameters, the transform can return the corresponding augmentation.


<a name="installation"></a>
## Installation
- To install the package directly, run the following commands:
```
git clone https://github.com/apple/parameterized-transforms
cd parameterized-transforms
pip install -e .
```
- To install the package via `pip`, run the following command:
```
pip install --upgrade https://github.com/apple/parameterized-transforms
```
- If you want to run unit tests locally, run the following steps:
```
git clone https://github.com/apple/parameterized-transforms
cd parameterized-transforms
pip install -e .
pip install -e '.[test]'
pytest
```


<a name="getting-started"></a>
## Getting Started
* To understand the structure of parameterized transforms and the details of the package, we recommend the reader to 
start with 
[The First Tutorial](https://apple.github.io/parameterized-transforms/tutorials/) 
of our
[Tutorial Series](https://apple.github.io/parameterized-transforms/).
* However, for a quick starter, check out [Parameterized Transforms in a Nutshell](https://pages.github.com/apple/parameterized-transforms/tutorials/999-In-a-Nutshell.md).

---

## Acknowledgement
In its development, this project received help from multiple researchers, engineers, and other contributors from Apple.
Special thanks to: Tim Kolecke, Jason Ramapuram, Russ Webb, David Koski, Mike Drob, Megan Maher Welsh, Marco Cuturi Cameto, 
Dan Busbridge, Xavier Suau Cuadros, and Miguel Sarabia del Castillo. 

## Citation
If you find this package useful and want to cite our work, here is the citation:
```
@software{Dhekane_Parameterized_Transforms_2025,
    author = {Dhekane, Eeshan Gunesh},
    month = {2},
    title = {{Parameterized Transforms}},
    url = {https://github.com/apple/parameterized-transforms},
    version = {1.0.0},
    year = {2025}
}
```

---

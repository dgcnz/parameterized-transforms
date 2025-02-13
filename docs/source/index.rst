Parameterized Transforms
========================


About the Package
-----------------

* The package provides a uniform, modular, and easily extendable implementation of `torchvision`-based transforms that provides access to their parameterization.

* With this access, the transforms enable users to achieve the following two important functionalities--

  * Given an image, return an augmentation along with the parameters used for the augmentation.

  * Given an image and augmentation parameters, return the corresponding augmentation.


Installation
------------
* To install the package directly, run the following commands:

.. code-block:: bash

   git clone git@github.com:apple/parameterized-transforms.git
   cd parameterized-transforms
   pip install -e .


* To install the package via `pip`, run the following command:

.. code-block:: bash

   pip install --upgrade git+https://git@github.com:apple/parameterized-transforms.git


Get Started
-----------
* To understand the structure of parameterized transforms and the details of the package, we recommend the reader to start with the :ref:`Tutorial Series <Tutorial-Series-label>`.

* Otherwise, for a quick starter, check out :ref:`Parameterized Transforms in a Nutshell <Quick-Start-label>`.


Important Links
---------------

.. _Quick-Start-label:

.. toctree::
    :caption: Quick-Start
    :maxdepth: 1

    tutorials/999-In-a-Nutshell

.. _Tutorial-Series-label:

.. toctree::
    :caption: Tutorial Series
    :maxdepth: 1

    tutorials/000-About-the-Package
    tutorials/001-The-Structure-of-Parametrized-Transforms
    tutorials/002-How-to-Write-Your-Own-Transforms
    tutorials/003-A-Brief-Introduction-to-the-Transforms-in-This-Package
    tutorials/004-Parametrized-Transforms-in-Action
    tutorials/005-Migrate-To-and-From-torch-in-Three-Easy-Steps


.. toctree::
    :caption: Python API Reference
    :maxdepth: 1

    python/core
    python/transforms
    python/utils
    python/wrappers

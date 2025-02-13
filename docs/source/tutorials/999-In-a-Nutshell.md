# Parameterized Transforms in a Nutshell


1. Parameterized transforms input an image and a tuple of parameters to produce augmentation and modified parameters.
The mode of any parameterized transform can be `CASCADE` or `CONSUME`.
In `CASCADE` mode, the transform generates an augmentation and appends parameters used in this augmentation to the input parameters.
In `CONSUME` mode, the transform removes from the beginning of the tuple the parameters it needs and generates the corresponding augmentation. 
The augmentation and the modified parameters are then returned.
With these two modes, we can have reproducible, flexible augmentations and much more. 
```python
# Signature of parameterized transforms.
import parameterized_transforms.transforms as ptx
import parameterized_transforms.core as ptc

tx1 = ptx.RandomRotation(degrees=45)  # Default mode: CASCADE.
params1 = (3, 2, 0, 1, 0.3, -2.5)  # Example: Parameters from previous parameterized transforms.

augmentation1, modified_params = tx1(image, params1)
# ALTERNATIVELY: augmentation1, modified_params = tx1.cascade_transform(image, params1)
# augmentation: Image rotated by a random angle, say 31.25 degrees.
# modified_params: (3, 2, 0, 1, 0.3, -2.5, 31.25). Note the appended 31.25 angle value.

tx2 = ptx.RandomRotation(degrees=45, tx_mode=ptc.TransformMode.CONSUME)
params2 = (31.25, 0, 1, 0.5, -1.7)  # Example: Parameter for `RandomRotation` (31.25) and possibly other parameterized transforms.
augmentation2, remaining_params = tx2(image, params2)
# ALTERNATIVELY: tx1.consume_transform(image, params2) or tx2.consume_transform(image, params2)
# augmentation2: The same augmentation as augmentation1 above.
# remaining_params: (0, 1, 0.5, -1.7).
```

2. Parameterized versions of all `torchvision`-based transforms are supported.
```python
# Example
import parameterized_transforms.transforms as ptx

tx = ptx.Compose(
    [
        ptx.RandomHorizontalFlip(p=0.5),
        ptx.RandomApply(
            [
                ptx.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.1,
                )
            ],
            p=0.5,
        ),
        ptx.ToTensor(),
        ptx.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

augmentation, params = tx(image)  # Default params: ().
augmentation_1, empty_params = tx.consume_transform(image, params)
# augmentation_1: Same as the augmentation above.
# empty_params: (), as all params are extracted and used.
```

3. You can [write your own transforms](https://pages.github.com/apple/parameterized-transforms/tutorials/002-How-to-Write-Your-Own-Transforms.html) that adhere to the structure of parameterized transforms.
Then, your transforms will work seamlessly with those from the package!
```python
import parameterized_transforms.transforms as ptx

tx = RandomSubsetApply(  # Your custom transform
    [
        RandomColorErasing(),  # Your custom transform   
        ptx.RandomHorizontalFlip(p=0.5),
        ptx.RandomApply(
            [
                ptx.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.1,
                )
            ],
            p=0.5,
        )
    ]
)

augmentation, params = tx(image)  # Default params: ().
augmentation_1, empty_params = tx.consume_transform(image, params)
# augmentation_1: Same as the augmentation above.
# empty_params: (), as all params are extracted and used.
```

4. You can use parameterized transforms with `torch`/`torchvision` dataset directly. 
However, in order to have parameters output as a single tensor of shape `[batch_size=B, num_params=P]`, we recommend wrapping your transform in `CastParamsToTensor` wrapper.
More on this in [tutorial on working with torch/torchvision](https://pages.github.com/apple/parameterized-transforms/tutorials/002-How-to-Write-Your-Own-Transforms.html).

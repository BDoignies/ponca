# Ponca C++ Binding

## Name mangling

Ponca C++ uses templates to provide generic yet performant code. However, templates are compile-time types and are therefore not directly compatible with Python. Instead, the bindings must instantiate and expose every template specialization that should be available to the user.

In Ponca, this is particularly challenging because the core type, `Point`, is itself a template parameter. Consequently, every class templated on `Point` would need to be bound under a different name. From the user's perspective, this would require using different classes depending on the scalar type or the dimension of the points, which is inconvenient.

To address this, we instead rely on name mangling and a Python-generated dispatcher. Within the C++ bindings, each class is registered under a unique mangled name. This name is computed from both the algorithm name (e.g., `APSS`) and the template type information (e.g., `Point<;float, 3>` becomes `3f`).

In `__init__.py`, we define a Python class that, once the required type information is known (i.e., when `compute()` is called), computes the mangled name of the corresponding C++ class and forwards all method calls to the appropriate implementation.


### Name mangling rules

We list here the rules for mangling. 

#### Array mangling rules

We encourage using the `MangleArray` function on the C++ side and `_pyponca.internal.mangleArray` on the Python side to ensure consistency between both implementations.

The first dimension of an array is never encoded in the mangled type. Subsequent dimensions are encoded as a list of their sizes, separated by underscores (`_`). For example, an array of shape `(12, 8, 4)` is encoded as `"8_4"`.

The array's scalar type is encoded according to the following mapping (using the first letter of the corresponding C++ type whenever possible):

* `float` → `"f"`
* `double` → `"d"`
* any other type → `"unknown"`

The shape and type encodings are then concatenated directly. For example, a `float` point cloud of shape `(N, 3)` has the mangled name `"3f"`.

#### Compute Object Mangling

Compute object names are constructed by concatenating the method name, the array mangling, and the point type mangling.

At the moment, users cannot select the `Filter` or `Diff` types, so these are not included in the mangled name. The method name and point type mangling are fixed strings defined in the factory and the C++ binding code, respectively.

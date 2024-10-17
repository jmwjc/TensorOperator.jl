# TensorOperator.jl

TensorOperator.jl is an efficient tool for performing tensor calculations.

## Features

- Utilizes Unicode symbols ⊗, ⋅, and × to represent the dyadic product, dot product, and cross product, respectively, and overloads Base.Colon for the double-dot product.
- Applicable to curvilinear coordinate systems with non-orthogonal basis vectors.
- High computational efficiency.

## Installation

```@setup 0
using TensorOperator
```

TensorOperator can be installed using the Julia package manager with the following commands:

```julia
using Pkg; Pkg.add("TensorOperator")
```

Alternatively, use the Pkg REPL mode:

```julia
pkg>add TensorOperator #Press `]` to enter the Pkg REPL mode.
```

## Usage

First, import TensorOperator with the using statement:

```@repl
using TensorOperator
```
### Construction

In TensorOperator, a `Tensor` is considered a tuple of `BaseTensor` objects. A `BaseTensor` is constructed from a basis vector (denoted as `base`) and its corresponding value (denoted as `component`). By default, in the 3D case, the basis tensors `e_i` (`i = 1,2,3`) are predefined as constants `e₁`, `e₂`, and `e₃`.

You can easily construct a tensor by combining these basis tensors, like so:

```@repl 0
u = 1e₁ + 2e₂ + 3e₃
```

For higher-order tensors, you can use the dyadic product (`⊗`) to build higher-order basis tensors, and then sum them to obtain the desired tensor:

```@repl 0
U = 4e₁⊗e₁ - 5e₂⊗e₃ + 6e₃⊗e₁
```

In this case, `U` consists of only three non-zero basis tensors. There are only three `BaseTensor` objects in `U`.

When adding two `BaseTensor` objects with the same base, the result will not combine their components directly. Instead, a `Tensor` containing both `BaseTensor` objects is created:

```@repl 0
V = 6e₁⊗e₂ + e₁⊗e₂ + 8e₂⊗e₃
```

Here, `V` contains three `BaseTensor` objects. However, for a tensor `W` defined as:

```@repl 0
W = 7e₁⊗e₂ + 8e₂⊗e₃
```

`V` and `W` will yield the same result in tensor calculations.

### Curvillinear system (non-orthogonal coordinates)

You can define custom basis tensors using `BaseTensor`. For example:

```@repl 0
g₁ = BaseTensor(1.0,2.0,3.0)
```

You can use this with other `BaseTensor` objects of the same dimension and order:

```@repl 0
v = 9e₁ + 10g₁
```

This approach makes constructing curvilinear systems straightforward.

### Projection

In TensorOperator.jl, tensors can be expressed with different base tensors. You can retrieve the components associated with specific base tensors.

By default, `Base.getindex` is overloaded to retrieve components corresponding to `eᵢ`:

```@repl 0
u[1]
U[2,3]
```

Even if the tensor is not part of the `eᵢ` base system, `Base.getindex` will return the projected component:

```@repl 0
v[3]
```

Alternatively, you can retrieve components corresponding to a custom coordinate system using `Base.getindex` with the bases keyword:

```@repl 0
g₂ = BaseTensor(4.0,5.0,6.0);
g₃ = BaseTensor(4.0,5.0,6.0);
u[2,bases = (g₁,g₂,g₃)]
```

### Operation examples

Here are some examples of tensor operations in TensorOperator.jl:

- Dyadic product

```@repl 0
u⊗v
```

- Dot product

```@repl 0
u⋅v
```

- Double product

```@repl 0
U:V
```

- Cross product

```@repl 0
u×v
```

- Trace

```@repl 0
tr(U)
```

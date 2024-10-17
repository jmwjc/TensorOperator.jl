# TensorOperator

[![Build Status](https://github.com/jmwjc/TensorOperator.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jmwjc/TensorOperator.jl/actions/workflows/CI.yml?query=branch%3Amain)

TensorOperator.jl is an efficient tool for performing tensor calculations.

## Features

- Utilizes Unicode symbols ⊗, ⋅, and × to represent the dyadic product, dot product, and cross product, respectively, and overloads Base.Colon for the double-dot product.
- Applicable to curvilinear coordinate systems with non-orthogonal basis vectors.
- High computational efficiency.

## Installation

TensorOperator can be installed using the Julia package manager with the following commands:

```julia
using Pkg; Pkg.add("TensorOperator")
```

Alternatively, use the Pkg REPL mode:

```julia
pkg>add TensorOperator #Press `]` to enter the Pkg REPL mode.
```

## Usage

In TensorOperator, a `Tensor` is considered a tuple of `BaseTensor` objects. A `BaseTensor` is constructed from a basis vector (denoted as `base`) and its corresponding value (denoted as `component`). By default, in the 3D case, the basis tensors `e_i` (`i = 1,2,3`) are predefined as constants `e₁`, `e₂`, and `e₃`.

You can easily construct a tensor by combining these basis tensors, like so:

```@repl 0
u = 1e₁ + 2e₂ + 3e₃
```

For more information on how to use this package, please refer to the [documentation](https://jmwjc.github.io/TensorOperator.jl/).

## To Do List

- [ ] Add benchmark examples for elasticity or shell models, and consider predefining special tensors, such as identity tensors, in the package.
- [ ] Provide examples demonstrating the implementation of function-like tensors.

## Related packages

The following packages also offer tensor calculations, and the motivation behind TensorOperator.jl is inspired by them. We would like to acknowledge them here:

- [Tensors.jl](https://github.com/Ferrite-FEM/Tensors.jl)
- [Einsum.jl](https://github.com/ahwillia/Einsum.jl)
- [TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl)
# TensorOperator.jl

TensorOperator.jl is an efficient tool to do the tensor calculation.

## Features

- Use the unicode symbols `⊗`, `⋅`, `×` to stands for the dyadic product, dot product, cross product respectively, and overloads `Base.Colon` as the double-dot product.
- Applicatable for curvilinear coordinate system that has the non-orthogonal basis vectors.
- High efficiency

## Installation

```@setup 0
using TensorOperator
```

TensorOperator can be installed using the Julia package manger, as follows

```julia
using Pkg; Pkg.add("TensorOperator")
```

or

```julia
pkg>add TensorOperator #Press `]` to enter the Pkg REPL mode.
```

## Usage

Firstly, TensorOperator should be introduced by `using` sentence.

```@repl
using TensorOperator
```

### Construction

In TensorOperator, a `Tensor` is regarded as a tuple of `BaseTensor`, a `BaseTensor` is constructed by a basis vector namely `base` and its corresponding value namely `component`.
By default, the basis tensor in 3D case,``e_i, i=1,2,3``, has been defined as a const `BaseTensor` e₁, e₂, e₃ in TensorOperator.
Accordingly, a tensor can be conveniently constructed by the combination of these basis tensor, like

```@repl 0
u = 1e₁ + 2e₂ + 3e₃
```

For high order tensor, you can use the dyadic product notion to build the high order basis tensor, and sum these to obtain a corresponding high order tensor

```@repl 0
U = 4e₁⊗e₁ - 5e₂⊗e₃ + 6e₃⊗e₁
```

It is noted that, for above case, `U` only has three basis tensors that their component is not zero, and there are only three `BaseTensor`'s in `U`.

A plus operation between two `BaseTensor` with same `base` is not means to combine their `component`'s. Instead, a `Tensor` with these two `BaseTensor` will be created, like

```@repl 0
V = 6e₁⊗e₂ + e₁⊗e₂ + 8e₂⊗e₃
```

where `V` acturally has three `BaseTensor` in it. However, for a `W` defined by follows

```@repl 0
W = 7e₁⊗e₂ + 8e₂⊗e₃
```

in this circumstance, `V` will shares same results with `W` in tensor calculation.

### Curvillinear system (non-orthogonal coordinates)

A specific base tensor can be defined via `BaseTensor`, like

```@repl 0
g₁ = BaseTensor(1.0,2.0,3.0)
```

and you can use it with other `BaseTensor`'s that has the same dimensions and orders, for example,

```@repl 0
v = 9e₁ + 10g₁
```

follow this path, the every coordiante systems, include curvillinear system, is easy to be contructed.

### Projection

From the previous examples, the tensor in TensorOperator.jl can be presented by different base tensors. And you can also get the componenet that has different base tensor.
By defaultly, the `Base.getindex` has been overload to get the component with eᵢ, like

```@repl 0
u[1]
U[2,3]
```

even the tensor does not belong the base tensor system of eᵢ's, the corresponding component after projection can also be got via `Base.getindex`, like

```@repl 0
v[3]
```

Otherwise, you can get any component that belongs to a custom coordinate system using `Base.getindex` with keyword `bases`.

```@repl 0
g₂ = BaseTensor(4.0,5.0,6.0);
g₃ = BaseTensor(4.0,5.0,6.0);
u[2,bases = (g₁,g₂,g₃)]
```

### Operation examples

The examples about the usage of some tensor calculations in TensorOperator.jl are list below

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

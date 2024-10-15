module TensorOperator

import Printf:@printf
import Base: +, -, *, Colon, sum, getindex, iterate, show
"""
    BaseTensor{dim,order,T}

Type for base tensors, where `dim` is the dimension, `order` is the tensor's order, `T` is the type for its component.
"""
struct BaseTensor{dim,order,T}
    component::T
    base::Tuple{Vararg{Tuple{Vararg{T,dim}},order}}
end

"""
    BaseTensor(e...)

The constructor for first-order base tensors with components `e`. The corresponding component is defaultly set to be `1.0`. Only first-order base tensors can be implemented by this function, i.e. `order = 1`, the high order ones can be constructed by dyadic operator `⊗`. The dimension of base tensor depends on the length of inputed components, i.e. `dim = length(e)`.
"""
BaseTensor(e...) = BaseTensor(1.0,(e,))

const e₁ = BaseTensor(1.0,0.0,0.0)
const e₂ = BaseTensor(0.0,1.0,0.0)
const e₃ = BaseTensor(0.0,0.0,1.0)

"""
    Tensor{dim,order,N,T}

Type for tensor, where `dim` is the dimension, `order` is tensor order, `N` is the number of base tensors in this tensor, `T` is the data type for component.
"""
struct Tensor{dim,order,N,T}
    bases::Tuple{Vararg{BaseTensor{dim,order,T},N}}
end
"""
    Tensor(e::BaseTensor...)

Constructor for tensor. Return a tenor with base tensors of `e`'s.
"""
Tensor(e::BaseTensor...) = Tensor(e)

"""
    *(c,e::BaseTensor)

Times operator for a constant and a base tensor. Return a BaseTensor with a component equal to `c` times the origin component.
"""
*(c,e::BaseTensor) = BaseTensor(c*e.component,e.base)

"""
    +(e::BaseTensor...)
    +(t::Tensor{dim,order},e::BaseTensor{dim,order}) where {dim,order}

Plus operator for tensors and base tensors, where the base tensors should have the same dimension `dim`, order `order` and component type `T`. Return a Tensor onstruced by base tensors `e`'s.
"""
+(e::BaseTensor...) = Tensor(e)
+(t::Tensor{dim,order},e::BaseTensor{dim,order}) where {dim,order} = Tensor(t.bases...,e)

"""
    -(t,e::BaseTensor)

Return `t` plus `e` that its component is negative.
"""
-(t,e::BaseTensor) = t + BaseTensor(-e.component,e.base)

"""
    sum(e::BaseTensor...)

Return a tensor with base tensors `e`.
"""
sum(e::BaseTensor...) = Tensor(e)

"""
    ⊗(e₁::BaseTensor{dim},e₂::BaseTensor{dim}) where dim
    ⊗(t::Tensor{dim},e::BaseTensor{dim}) where dim
    ⊗(e::BaseTensor{dim},t::Tensor{dim}) where dim
    ⊗(t₁::Tensor{dim,order_1,N_1},t₂::Tensor{dim,order_2,N_2}) where {dim,order_1,order_2,N_1,N_2}

Dyadic operaotrs for tensors and base tensors. Inputing two base tensors returns a high order base tensor. Inputing at least one tensor returns a tensor with high order base tensors.
"""
⊗(e₁::BaseTensor{dim},e₂::BaseTensor{dim}) where dim = BaseTensor(e₁.component*e₂.component,(e₁.base...,e₂.base...))
⊗(t::Tensor{dim},e::BaseTensor{dim}) where dim = sum(eᵢ⊗e for eᵢ in t.bases)
⊗(e::BaseTensor{dim},t::Tensor{dim}) where dim = sum(e⊗eᵢ for eᵢ in t.bases)
⊗(t₁::Tensor{dim,order_1,N_1},t₂::Tensor{dim,order_2,N_2}) where {dim,order_1,order_2,N_1,N_2} = Tensor(Tuple{Vararg{BaseTensor{dim,order_1+order_2},N_1*N_2}}(t₁.bases[i]⊗t₂.bases[j] for (i,j) in Iterators.product(1:N_1,1:N_2)))

"""
    ⋅(e₁::BaseTensor{dim},e₂::BaseTensor{dim}) where dim
    ⋅(t::Tensor{dim},e::BaseTensor{dim}) where dim
    ⋅(e::BaseTensor{dim},t::Tensor{dim}) where dim
    ⋅(t₁::Tensor{dim,order_1,N_1},t₂::Tensor{dim,order_2,N_2}) where {dim,order_1,order_2,N_1,N_2}

Dot product for tensors and base tensors. The operations carried between first-order tensors or base tensors will return a scale value. Other operations will return a low order tensor or base tensor.
"""
⋅(e₁::BaseTensor{dim,1},e₂::BaseTensor{dim,1}) where dim = e₁.component*e₂.component*sum(map(*,e₁.base[1],e₂.base[1]))
⋅(e₁::BaseTensor{dim},e₂::BaseTensor{dim}) where dim = BaseTensor(e₁.component*e₂.component*sum(map(*,e₁.base[end],e₂.base[1])),(e₁.base[1:end-1]...,e₂.base[2:end]...))
⋅(t::Tensor{dim},e::BaseTensor{dim}) where dim = sum(eᵢ⋅e for eᵢ in t.bases)
⋅(e::BaseTensor{dim},t::Tensor{dim}) where dim = sum(e⋅eᵢ for eᵢ in t.bases)
⋅(t₁::Tensor{dim,order_1,N_1},t₂::Tensor{dim,order_2,N_2}) where {dim,order_1,order_2,N_1,N_2} = Tensor(Tuple{Vararg{BaseTensor{dim,order_1+order_2-2},N_1*N_2}}(t₁.bases[i]⋅t₂.bases[j] for (i,j) in Iterators.product(1:N_1,1:N_2)))
⋅(t₁::Tensor{dim,1},t₂::Tensor{dim,1}) where dim = sum(e₁⋅e₂ for (e₁,e₂) in Iterators.product(t₁.bases,t₂.bases))

"""
    (::Colon)(e₁::BaseTensor{dim},e₂::BaseTensor{dim}) where dim
    (::Colon)(t::Tensor{dim},e::BaseTensor{dim}) where dim
    (::Colon)(e::BaseTensor{dim},t::Tensor{dim}) where dim
    (::Colon)(t₁::Tensor{dim,order_1,N_1},t₂::Tensor{dim,order_2,N_2}) where {dim,order_1,order_2,N_1,N_2}

Double-dot product for tensors and base tensors. These operaotrs only suit for the tensors that their order greater than or equal to 2. The operations between second-order tensors or base tensors will return a scale value. Other operations will return a low order tensor or base tensor.
"""
(::Colon)(e₁::BaseTensor{dim,2},e₂::BaseTensor{dim,2}) where dim = e₁.component*e₂.component*sum(map(*,e₁.base[1],e₂.base[1]))*sum(map(*,e₁.base[2],e₂.base[2]))
(::Colon)(e₁::BaseTensor{dim},e₂::BaseTensor{dim}) where dim = BaseTensor(e₁.component*e₂.component*sum(map(*,e₁.base[end-1],e₂.base[1]))*sum(map(*,e₁.base[end],e₂.base[2])),(e₁.base[1:end-2]...,e₂.base[3:end]...))
(::Colon)(t::Tensor{dim},e::BaseTensor{dim}) where dim = sum(eᵢ:e for eᵢ in t.bases)
(::Colon)(e::BaseTensor{dim},t::Tensor{dim}) where dim = sum(e:eᵢ for eᵢ in t.bases)
(::Colon)(t₁::Tensor{dim,2},t₂::Tensor{dim,2}) where dim = sum(e₁:e₂ for (e₁,e₂) in Iterators.product(t₁.bases,t₂.bases))
(::Colon)(t₁::Tensor{dim,order_1,N_1},t₂::Tensor{dim,order_2,N_2}) where {dim,order_1,order_2,N_1,N_2} = Tensor(Tuple{Vararg{BaseTensor{dim,order_1+order_2-4},N_1*N_2}}(e₁:e₂ for (e₁,e₂) in Iterators.product(t₁.bases,t₂.bases)))

"""
    ×(v₁::Tuple{Vararg{T,3}},v₂::Tuple{Vararg{T,3}}) where T
    ×(e₁::BaseTensor{3},e₂::BaseTensor{3})
    ×(t::Tensor{dim},e::BaseTensor{dim}) where dim
    ×(e::BaseTensor{dim},t::Tensor{dim}) where dim
    ×(t₁::Tensor{dim,order_1,N_1},t₂::Tensor{dim,order_2,N_2}) where {dim,order_1,order_2,N_1,N_2}

Cross product for tensors and base tensors. These operaotrs only suit for the tensors that their order greater than or equal to 2. The operations between second-order tensors or base tensors will return a scale value. Other operations will return a low order tensor or base tensor.
"""
function ×(v₁::Tuple{Vararg{T,3}},v₂::Tuple{Vararg{T,3}}) where T
    v₃₁ = v₁[2]*v₂[3]-v₁[3]*v₂[2]
    v₃₂ = v₁[3]*v₂[1]-v₁[1]*v₂[3]
    v₃₃ = v₁[1]*v₂[2]-v₁[2]*v₂[1]
    return (v₃₁,v₃₂,v₃₃)
end
×(e₁::BaseTensor{3},e₂::BaseTensor{3}) = BaseTensor(e₁.component*e₂.component,(e₁.base[1:end-1]...,e₁.base[end]×e₂.base[1],e₂.base[2:end]...))
×(t::Tensor{dim},e::BaseTensor{dim}) where dim = sum(eᵢ×e for eᵢ in t.bases)
×(e::BaseTensor{dim},t::Tensor{dim}) where dim = sum(e×eᵢ for eᵢ in t.bases)
×(t₁::Tensor{dim,order_1,N_1},t₂::Tensor{dim,order_2,N_2}) where {dim,order_1,order_2,N_1,N_2} = Tensor(Tuple{Vararg{BaseTensor{dim,order_1+order_2-1},N_1*N_2}}(t₁.bases[i]×t₂.bases[j] for (i,j) in Iterators.product(1:N_1,1:N_2)))

"""
    tr(t::Tensor{3,2})

Return the trace of tensor `t`. This function is only applicatable for the tensor with `dim = 3` and `order = 2`.
"""
tr(t::Tensor{3,2}) = sum(t[i,i] for i in 1:3)

"""
    project(t::Tensor{dim},bases_tensor::Tuple{Vararg{BaseTensor{dim,1},dim}}=(e₁,e₂,e₃)) where dim

Project `t` to a tensor with the base tensor `bases` and return it. This function can also be used to simplfy `t`'s base tensors to be independent.
"""
function project(t::Tensor{dim,1,N},bases::Tuple{Vararg{BaseTensor{dim,1},dim}}=(e₁,e₂,e₃)) where {dim,N}
    components = (sum(eᵢ⋅eⱼ for eᵢ in t.bases) for eⱼ in bases)
    return Tensor(Tuple{Vararg{BaseTensor{dim,1},dim}}(component*base for (component,base) in zip(components,bases)))
end

function project(t::Tensor{dim,2},bases_tensor::Tuple{Vararg{BaseTensor{dim,1},dim}}=(e₁,e₂,e₃)) where dim
    N = dim^2
    bases = Tuple{Vararg{BaseTensor{dim,2},N}}(bases_tensor[i]⊗bases_tensor[j] for (i,j) in Iterators.product(1:dim,1:dim))
    components = Tuple{Vararg{Float64,N}}(sum(eᵢ:eⱼ for eᵢ in t.bases) for eⱼ in bases)
    return Tensor(Tuple{Vararg{BaseTensor{dim,2},N}}(component*base for (component,base) in zip(components,bases)))
end

function project(t::Tensor{dim,order},bases_tensor::Tuple{Vararg{BaseTensor{dim,1},dim}}=(e₁,e₂,e₃)) where {dim,order}
    N = dim^order
    indices = Iterators.product(ntuple(_->1:dim,order)...)
    bases = (reduce(⊗,bases_tensor[i] for i in index) for index in indices)
    # components = (t[index...] for index in indices for index in indices)
    # bases = Tuple{Vararg{BaseTensor{dim,order},N}}(reduce(⊗,e) for e in Iterators.product(ntuple(_->bases_tensor,order)...))
    components = Tuple{Vararg{Float64,N}}(sum(ndot(eᵢ,eⱼ) for eᵢ in t.bases) for eⱼ in bases)
    return Tensor(Tuple{Vararg{BaseTensor{dim,order},N}}(component*base for (component,base) in zip(components,bases)))
end

ndot(e₁::BaseTensor{dim,order},e₂::BaseTensor{dim,order}) where {dim,order} = Float64(e₁.component*e₂.component*reduce(*,(map((t₁,t₂)->sum(map(*,t₁,t₂)),e₁.base,e₂.base))))

function getindex(t::Tensor{dim,order,N,T},I::Int...;bases::Tuple{Vararg{BaseTensor{dim,1},dim}} = (e₁,e₂,e₃)) where {dim,order,N,T}
    e = reduce(⊗,bases[i] for i in I)
    return sum(ndot(eᵢ,e) for eᵢ in t.bases)::T
end

function print_base_tensor(io::IO,e::BaseTensor{dim,order},i::Int,j::Int) where {dim,order}
    m = ceil(Int,dim/2)
    if j ≠ 1
        if i ≠ m
            @printf io " "
        else
            @printf io "⊗"
        end
    end
    if i == 1
        @printf io "⎛%5.2f⎞" e.base[j][1]
    elseif i == dim
        @printf io "⎝%5.2f⎠" e.base[j][dim]
    else
        @printf io "⎜%5.2f⎟" e.base[j][i]
    end
end

function show(io::IO,e::BaseTensor{dim,order}) where {dim,order}
    @printf io "BaseTensor(Dim = %d, Order = %d):\n" dim order
    for i in 1:dim
        for j in 1:order
            print_base_tensor(io,e,i,j)
        end
        @printf io "\n"
    end
end
function show(io::IO,::MIME"text/plain",e::BaseTensor{dim,order}) where {dim,order}
    @printf io "BaseTensor(Dim = %d, Order = %d):\n" dim order
    for i in 1:dim
        for j in 1:order
            print_base_tensor(io,e,i,j)
        end
        @printf io "\n"
    end
end
function show(io::IO,::MIME"text/latex",e::BaseTensor{dim,order}) where {dim,order}
    print(io,"\$\$ y=kx+b \$\$")
end

function show(io::IO,t::Tensor{dim,order,N,T}) where {dim,order,N,T}
    @printf io "Tensor(Dim = %d, Order = %d):\n" dim order
    for (i,e) in enumerate(t.bases)
        i ≠ N ? (@printf io "  ├─ %f " e.component) : (@printf io "  └─ %f " e.component)
        print(io,e.base[1])
        for v in e.base[2:end]
            print(io,"⊗")
            print(io,v)
        end
        @printf io "\n"
    end
end
function show(io::IO,::MIME"text/plain",t::Tensor{dim,order,N,T}) where {dim,order,N,T}
    @printf io "Tensor(Dim = %d, Order = %d):\n" dim order
    for (i,e) in enumerate(t.bases)
        i ≠ N ? (@printf io "  ├─ %f " e.component) : (@printf io "  └─ %f " e.component)
        print(io,e.base[1])
        for v in e.base[2:end]
            print(io,"⊗")
            print(io,v)
        end
        @printf io "\n"
    end
end

export Tensor, BaseTensor, e₁, e₂, e₃
export ⊗, ⋅, Colon, ×, tr, project

end

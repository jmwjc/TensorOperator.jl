using Revise
using TensorOperator
using BenchmarkTools, Profile
import Base: getindex

g₁ = BaseTensor(0.5,0.5,0.0)
g₂ = BaseTensor(0.5,0.0,0.5)
g₃ = BaseTensor(0.0,0.5,0.5)
bases = (g₁,g₂,g₃)

h₁₁ = g₁⊗g₁
h₁₂ = g₁⊗g₂

a₁ = 1.0g₁ + 2.0g₂ + 3.0g₃
a₂ = 1.0g₁⊗g₁ + 2.0g₁⊗g₂ + 3.0g₁⊗g₃
a₃ = 1.0g₁⊗g₁⊗g₁ + 2.0g₁⊗g₂⊗g₂ + 3.0g₁⊗g₂⊗g₃
a₄ = 1.0g₁⊗g₁⊗g₁⊗g₁ + 2.0g₁⊗g₂⊗g₂⊗g₂ + 3.0g₁⊗g₂⊗g₃⊗g₃

println("======== Test for BaseTensor ========")
print("Construction 1:")
@btime 1.0*$g₁ + 2.0*$g₂ + 3.0*$g₃
print("Construction 2:")
@btime sum(($g₁,$g₂,$g₃))
print("Dyad product (⊗):")
@btime $g₁⊗$g₂
print("Dot product (⋅):")
@btime $g₁⋅$g₂
print("Double-dot product (:)")
@btime $h₁₁:$h₁₂
print("Cross product (×):")
@btime $g₁×$g₂

println("======== Test for Tensor ========")
print("Dyad product with BaseTensor (⊗):")
@btime $a₁⊗$g₂
print("Dot product with BaseTensor (⋅):")
@btime $a₁⋅$g₂
print("Double-dot product with BaseTensor (:)")
@btime $a₂:$h₁₂
print("Cross product with BaseTensor (×):")
@btime $a₂×$g₂
print("Dyad product (⊗):")
@btime $a₁⊗$a₂
print("Dot product (⋅):")
@btime $a₁⋅$a₂
print("Double-dot product (:)")
@btime $a₂:$a₂
print("Cross product (×):")
@btime $a₂×$a₂
print("Projection for first-order tensor")
@btime project($a₁)
print("Projection for second-order tensor")
@btime project($a₂)
print("Projection for third-order tensor")
@btime project($a₃)
print("Projection for fourth-order tensor")
@btime project($a₄)
print("Getindex for first-order tensor")
@btime $a₁[1]
print("Getindex for second-order tensor")
@btime $a₂[1,2]
print("Getindex for second-order tensor with specific bases")
@btime Base.getindex($a₂,1,2,bases=($g₁,$g₂,$g₃))
# @code_warntype Base.getindex(a₂,1,2,bases=(g₁,g₂,g₃))

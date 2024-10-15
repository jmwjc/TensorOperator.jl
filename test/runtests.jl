using TensorOperator
using Test

@testset "TensorOperator.jl" begin
    g₁ = BaseTensor(0.5,0.5,0.0)
    g₂ = BaseTensor(0.5,0.0,0.5)
    g₃ = BaseTensor(0.0,0.5,0.5)

    g¹ = BaseTensor(-0.25,-0.25,0.25)
    g² = BaseTensor(-0.25,0.25,-0.25)
    g³ = BaseTensor(0.25,-0.25,-0.25)

    h₁₁ = g₁⊗g₁
    h₁₂ = g₁⊗g₂

    g₁₁ = g₁⋅g₁
    g₁₂ = g₁⋅g₂
    g₁₃ = g₁⋅g₃
    g₂₂ = g₂⋅g₂
    g₂₃ = g₂⋅g₃
    g₃₃ = g₃⋅g₃

    bases = (g₁,g₂,g₃)

    a₁ = 1.0g₁ + 2.0g₂ + 3.0g₃
    a₂ = 1.0g₁⊗g₁ + 2.0g₁⊗g₂ + 3.0g₁⊗g₃
    a₃ = 1.0g₁⊗g₁⊗g₁ + 2.0g₁⊗g₂⊗g₂ + 3.0g₁⊗g₂⊗g₃
    a₄ = 1.0g₁⊗g₁⊗g₁⊗g₁ + 2.0g₁⊗g₂⊗g₂⊗g₂ + 3.0g₁⊗g₂⊗g₃⊗g₃

    ā₁ = project(a₁)
    ā₂ = project(a₂)
    ā₃ = project(a₃)

    I = rand(1:3)
    J = rand(1:3)
    K = rand(1:3)

    # construction
    @test a₁ == sum(1.0g₁,2.0g₂,3.0g₃)
    @test a₂ == Tensor(1.0g₁⊗g₁,2.0g₁⊗g₂,3.0g₁⊗g₃)

    # daydic product
    @test g₁⊗g₂ == BaseTensor(1.0,((0.5,0.5,0.0),(0.5,0.0,0.5)))
    @test a₁⊗g₂ == g₁⊗g₂ + 2.0g₂⊗g₂ + 3.0g₃⊗g₂
    @test a₁⊗a₂ == 1.0g₁⊗g₁⊗g₁ + 2.0g₂⊗g₁⊗g₁ + 3.0g₃⊗g₁⊗g₁ + 2.0g₁⊗g₁⊗g₂ + 4.0g₂⊗g₁⊗g₂ + 6.0g₃⊗g₁⊗g₂ + 3.0g₁⊗g₁⊗g₃ + 6.0g₂⊗g₁⊗g₃ + 9.0g₃⊗g₁⊗g₃

    # dot product
    @test g₁⋅g₂ == 0.25
    @test a₁⋅g₂ == 2.0
    @test a₁⋅a₂ == 1.0*g₁₁*g₁ + 2.0*g₁₂*g₁ + 3.0*g₁₃*g₁ + 2.0*g₁₁*g₂ + 4.0*g₁₂*g₂ + 6.0*g₁₃*g₂ + 3.0*g₁₁*g₃ + 6.0*g₁₂*g₃ + 9.0*g₁₃*g₃

    # double-dot product
    @test h₁₁:h₁₂ == 0.125
    @test a₂:h₁₂ == 1.0
    @test a₂:a₂ == g₁₁*g₁₁ + 4.0*g₁₁*g₂₂ + 9.0*g₁₁*g₃₃ + 4.0*g₁₁*g₁₂ + 6.0*g₁₁*g₁₃ + 12.0*g₁₁*g₂₃

    # cross product
    @test g₁×g₂ == BaseTensor(0.25,-0.25,-0.25)
    @test (a₂×g₂)[I,J] == (1.0g₁⊗g³ - 3.0g₁⊗g¹)[I,J]
    @test (a₂×a₂)[I,J,K] == (- 2.0*g₁⊗g³⊗g₁ - 4.0*g₁⊗g³⊗g₂ - 6.0*g₁⊗g³⊗g₃ + 3.0*g₁⊗g²⊗g₁ + 6.0*g₁⊗g²⊗g₂ + 9.0*g₁⊗g²⊗g₃)[I,J,K]

    # project and getindex
    @test ā₁[I] == a₁[I]
    @test ā₂[I,J] == a₂[I,J]
    @test ā₃[I,J,K] == a₃[I,J,K]
end

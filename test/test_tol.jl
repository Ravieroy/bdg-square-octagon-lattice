using Test

@testset "check_rel_tol" begin

    @testset "Real values – pass" begin
        old = [1.0, 2.0, 3.0]
        new = [1.0 + 1e-7, 2.0 - 1e-7, 3.0 + 1e-7]
        @test check_rel_tol(old, new; tol=1e-5)
    end

    @testset "Real values – fail" begin
        old = [1.0, 2.0, 3.0]
        new = [1.0, 2.0, 3.001]
        @test !check_rel_tol(old, new; tol=1e-5)
    end

    @testset "Complex values" begin
        old = [1.0 + 1im, 2.0 - 1im]
        new = old .+ 1e-7
        @test check_rel_tol(old, new)
    end

    @testset "Length mismatch throws" begin
        old = [1.0, 2.0]
        new = [1.0]
        @test_throws AssertionError check_rel_tol(old, new)
    end

    @testset "Mixed-scale values" begin
        old = [1.0, 1e-3, 1e3]
        new = [
            1.0   + 1e-7,   # very small relative change
            1e-3  + 1e-8,   # small relative change
            1e3   + 1e-1    # relative change = 1e-4 (too large)
        ]

        @test !check_rel_tol(old, new; tol=1e-5)
    end

    @testset "Mixed-scale values – pass" begin
        old = [1.0, 1e-6, 1e6]
        new = old .* (1 .+ 1e-7)

        @test check_rel_tol(old, new; tol=1e-5)
    end

end

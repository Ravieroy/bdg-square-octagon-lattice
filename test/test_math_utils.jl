using Test
using QuadGK

@testset "kronecker_delta" begin

    @testset "Basic equality" begin
        @test kronecker_delta(1, 1) == 1
        @test kronecker_delta(2, 2) == 1
    end

    @testset "Basic inequality" begin
        @test kronecker_delta(1, 2) == 0
        @test kronecker_delta(3, 1) == 0
    end

    @testset "Identity matrix property" begin
        n = 5
        for i in 1:n, j in 1:n
            expected = (i == j) ? 1 : 0
            @test kronecker_delta(i, j) == expected
        end
    end

    @testset "Type stability" begin
        @test kronecker_delta(1, 1) isa Int
        @test kronecker_delta(1, 2) isa Int
    end

end


@testset "delta_fn" begin

    @testset "Symmetry" begin
        Γ = 0.1
        @test delta_fn(0.5, Γ) ≈ delta_fn(-0.5, Γ)
    end

    @testset "Peak at zero" begin
        Γ = 0.2
        @test delta_fn(0.0, Γ) ≈ inv(pi * Γ)
    end

    @testset "Decay away from zero" begin
        Γ = 0.1
        @test delta_fn(0.0, Γ) > delta_fn(0.5, Γ)
        @test delta_fn(0.5, Γ) > delta_fn(1.0, Γ)
    end

    @testset "Finite values" begin
        Γ = 0.05
        for x in (-10.0, -1.0, 0.0, 1.0, 10.0)
            val = delta_fn(x, Γ)
            @test isfinite(val)
            @test val ≥ 0
        end
    end

    @testset "delta_fn normalization" begin
        Γ = 0.1
        f(x) = delta_fn(x, Γ)

        val, err = quadgk(f, -Inf, Inf)

        @test isapprox(val, 1.0; atol=1e-6)
    end

end


@testset "fermi_fn" begin

    @testset "Half filling" begin
        @test isapprox(fermi_fn(0.0; T=1.0), 0.5; atol=1e-12)
        @test isapprox(fermi_fn(0.0, 0.0; beta=2.0), 0.5; atol=1e-12)
    end

    @testset "Low temperature limits" begin
        T = 0.01
        @test fermi_fn(1.0; T=T) < 1e-6
        @test fermi_fn(-1.0; T=T) > 1 - 1e-6
    end

    @testset "Symmetry around mu" begin
        T = 0.5
        E = 0.7
        @test isapprox(
            fermi_fn(E; T=T),
            1 - fermi_fn(-E; T=T);
            atol=1e-12
        )
    end

    @testset "Beta and T equivalence" begin
        E = 0.3
        T = 0.4
        beta = inv(T)
        @test isapprox(
            fermi_fn(E; T=T),
            fermi_fn(E; beta=beta);
            atol=1e-12
        )
    end

    @testset "Finite and bounded" begin
        for E in (-100.0, -1.0, 0.0, 1.0, 100.0)
            val = fermi_fn(E; T=1.0)
            @test isfinite(val)
            @test 0.0 <= val <= 1.0
        end
    end

    @testset "Missing arguments" begin
        @test_throws ArgumentError fermi_fn(1.0)
    end

end


@testset "fermi_fn_classic" begin

    @testset "Half filling" begin
        @test isapprox(fermi_fn_classic(0.0, 1.0), 0.5; atol=1e-12)
    end

    @testset "Low temperature limits" begin
        T = 0.01
        @test fermi_fn_classic(1.0, T) < 1e-6
        @test fermi_fn_classic(-1.0, T) > 1 - 1e-6
    end

    @testset "Agreement with tanh form" begin
        for T in (0.1, 0.5, 1.0)
            for E in (-2.0, -0.5, 0.0, 0.5, 2.0)
                @test isapprox(
                    fermi_fn(E; T=T),
                    fermi_fn_classic(E, T);
                    atol=1e-10
                )
            end
        end
    end

    @testset "Finite and bounded" begin
        for E in (-100.0, -1.0, 0.0, 1.0, 100.0)
            val = fermi_fn_classic(E, 1.0)
            @test isfinite(val)
            @test 0.0 <= val <= 1.0
        end
    end

end


@testset "is_symmetric" begin

    @testset "Basic symmetric matrix" begin
        A = [1.0 2.0;
             2.0 3.0]
        @test is_symmetric(A)
    end

    @testset "Non-symmetric matrix" begin
        A = [1.0 2.0;
             3.0 4.0]
        @test !is_symmetric(A)
    end

    @testset "Tolerance handling" begin
        A = [1.0 2.0;
             2.0 3.0 + 1e-12]
        @test is_symmetric(A; tol=1e-10)
    end

    @testset "Reject complex matrices" begin
        A = [1+0im 2+im;
             2-im  3+0im]
        @test_throws MethodError is_symmetric(A)
    end

    @testset "Non-square matrix" begin
        A = rand(2, 3)
        @test_throws ArgumentError is_symmetric(A)
    end

end


@testset "is_hermitian" begin

    @testset "Real symmetric matrix" begin
        A = [1.0 2.0;
             2.0 3.0]
        @test is_hermitian(A)
    end

    @testset "Complex Hermitian matrix" begin
        A = [1+0im  2+im;
             2-im   3+0im]
        @test is_hermitian(A)
    end

    @testset "Complex non-Hermitian matrix" begin
        A = [1+0im  2+im;
             2+im   3+0im]
        @test !is_hermitian(A)
    end

    @testset "Tolerance handling" begin
        A = [1+0im  2+im;
             2-im   3+1e-12]
        @test is_hermitian(A; tol=1e-10)
    end

    @testset "Non-square matrix" begin
        A = rand(3, 2)
        @test_throws ArgumentError is_hermitian(A)
    end

end


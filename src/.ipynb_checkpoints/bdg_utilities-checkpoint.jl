module BdGUtilities
    using LatticeUtilities
    using DelimitedFiles
    using LinearAlgebra
    using Plots
    using DataFrames
    using CSV
    using LaTeXStrings
    using Statistics
    using PyCall
    @pyimport numpy as np

    include("../src/general_utils.jl")
    using .GeneralUtils

    # Module exports
    export sort_evecs
    export check_norm
    export generate_t_matrix
    export calc_delta
    export calc_avg_n
    export create_bdg_ham_up
    export create_bdg_ham_dn
    export run_self_consistency_numpy

    # alias to fermi function
    const f = fermi_fn

    """
    sort_evecs(evector::AbstractMatrix{T}, nSites::Int) where T <: Union{Float64, ComplexF64}

    Sorts and slices an eigenvector matrix `evector` to separate components `uₙ` and `vₙ`.

    # Arguments
    - `evector::AbstractMatrix{T}`: Matrix of eigenvectors (can be real or complex).
    - `nSites::Int`: Number of lattice sites.

    # Returns
    - Tuple of matrices (`uₙ`, `vₙ`), each with element type `T`.
    """
    function sort_evecs(evector::AbstractMatrix{T}, nSites::Int) where T <: Union{Float64, ComplexF64}
        # Transpose to rearrange the vectors for easier slicing
        evecs = transpose(evector)[nSites + 1 : 2 * nSites, :]

        uₙ = evecs[:, 1:nSites]
        vₙ = evecs[:, nSites + 1:2 * nSites]

        return uₙ, vₙ
    end

    """
    check_norm(uₙ::AbstractMatrix{T}, vₙ::AbstractMatrix{T}, nSites::Int; digits::Int=5) where T <: Union{Float64, ComplexF64}

    Checks the normalization condition |uₙ|^2 + |vₙ|^2 = 1 at every site for
    each eigenvalue `n`, with optional rounding precision.

    # Arguments
    - `uₙ::AbstractMatrix{T}`: Matrix representing the `u` component.
    - `vₙ::AbstractMatrix{T}`: Matrix representing the `v` component.
    - `nSites::Int`: Number of lattice sites.
    - `digits::Int`: Number of decimal places to round the output to (default is 5).

    # Returns
    - `normList::Vector{Float64}`: A vector of norm values rounded to the specified precision.
    """
    function check_norm(uₙ::AbstractMatrix{T}, vₙ::AbstractMatrix{T}, nSites::Int; digits::Int=5) where T <: Union{Float64, ComplexF64}
        normList = zeros(Float64, nSites)
        for n in 1:nSites
            val = dot(uₙ[n, :], uₙ[n, :]) + dot(vₙ[n, :], vₙ[n, :])
            normList[n] = val
        end
        return round.(normList, digits=digits)
    end


    function generate_t_matrix(fileName::String)::Matrix{Complex{Float64}}
        rawHam = Matrix{Float64}(readdlm(fileName))

        nrows = size(rawHam, 1)
        ncols = size(rawHam, 1)

        H = zeros(Complex{Float64}, nrows, ncols)

        for row in 1:nrows
            c = 1
            for col in 1:ncols
                # Assign real and imaginary parts
                H[row, col] = rawHam[row, c] + im * rawHam[row, c+1]
                c += 2 # Increment by 2 for pairs (real, imaginary)
            end
        end
        return H
    end


    """
        calc_delta(U, un_up, vn_up, un_dn, vn_dn, nSites, E₁, E₂, T)

    Calculates the superconducting gap Δ for each site.

    # Arguments
    - `U`: Interaction parameter.
    - `un_up`: `u` component for spin-up.
    - `vn_up`: `v` component for spin-up.
    - `un_dn`: `u` component for spin-down.
    - `vn_dn`: `v` component for spin-down.
    - `nSites`: Number of lattice sites.
    - `E₁`: Eigenvalues for the first set of states.
    - `E₂`: Eigenvalues for the second set of states.
    - `T`: Temperature.

    # Returns
    - `ΔList`: List of calculated Δ values for each site, which can be of type Float64 or ComplexF64.
    """
    function calc_delta(U, un_up, vn_up, un_dn, vn_dn, nSites, E₁, E₂, T)
        ΔList = zeros(ComplexF64, nSites)
        for i in 1:nSites
            Δ = zero(ComplexF64)
            for n in 1:nSites
                E₁ₙ = E₁[n]
                E₂ₙ = E₂[n]
                fₙ₁ = f(E₁ₙ, T=T)
                fₙ₂ = f(E₂ₙ, T=T)
                Δ += U * ((un_up[n, i] * conj(vn_dn[n, i])) * (1 - fₙ₁) - (un_dn[n, i] * conj(vn_up[n, i])) * fₙ₂)
            end
            ΔList[i] = Δ
        end

        # Convert to Float64 if all values are real
        if all(isreal, ΔList)
            return Float64.(ΔList)
        end

        return ΔList
    end


    """
        calc_delta(U, un_up, vn_up, un_dn, vn_dn, nSites, E₁, E₂, T)

    Calculates the superconducting gap Δ for each site.

    # Arguments
    - `un`: `u` component for spin-up.
    - `vn`: `v` component for spin-up.
    - `nSites`: Number of lattice sites.
    - `E₁`: Eigenvalues for the first set of states.
    - `E₂`: Eigenvalues for the second set of states.
    - `T`: Temperature.

    # Returns
    - `n̄List`: List of calculated average value of number density of electrons(n̄) for each site.
    """
    function calc_avg_n(un, vn, nSites, E₁, E₂, T)
        n̄List = zeros(Float64, nSites)
        for i in 1:nSites
            n̄ = 0
            for n in 1:nSites
                E₁ₙ = E₁[n]
                E₂ₙ = E₂[n]
                fₙ₁ = f(E₁ₙ, T=T)
                fₙ₂ = f(E₂ₙ, T=T)
                n̄ += (abs2(un[n, i]) * fₙ₁ + abs2(vn[n, i]) * (1 - fₙ₂))
            end
            n̄List[i] = n̄
        end
        return n̄List
    end


    function create_bdg_ham_up(deltaList, H, μ, nSites, n_up, n_dn, U, impuritySite, J; K=0, isComplex=false)
        # Spin up and spin down indices
        σ_up = 1    # Spin-up
        σ_dn = -1   # Spin-down

        # Initialize the BdG Hamiltonian
        if isComplex
            HBdG = zeros(Complex{Float64}, 2 * nSites, 2 * nSites)
        else
            HBdG = zeros(Float64, 2 * nSites, 2 * nSites)
        end

        #----- Upper-left block: Spin-up sector------
        for i in 1:nSites
            if i == impuritySite
                H[i, i] = -(μ - (U * n_dn[i]) - (K - σ_up * J))
            else
                H[i, i] = -(μ - (U * n_dn[i]))
            end
        end

        for i in 1:nSites
            for j in 1:nSites
                HBdG[i, j] = H[i, j]
                HBdG[nSites + i, nSites + j] = -conj(H[i, j])
            end
        end

        # Add pairing terms D_ij in the off-diagonal
        for i in 1:nSites
            HBdG[i, nSites + i] = deltaList[i]          # D_ij in upper-left block
            HBdG[nSites + i, i] = conj(deltaList[i])    # D_ij^* in the upper-left block
        end

        #----- Upper-left block: Spin-down sector------
        for i in 1:nSites
            if i == impuritySite
                HBdG[nSites + i, nSites + i] = (μ - (U * n_up[i]) - (K - σ_dn * J))
            else
                HBdG[nSites + i, nSites + i] = (μ - (U * n_up[i]))
            end
        end
        return HBdG
    end


    function create_bdg_ham_dn(deltaList, H, μ, nSites, n_up, n_dn, U, impuritySite, J; K=0, isComplex=false)
        # Spin up and spin down indices
        σ_up = 1    # Spin-up
        σ_dn = -1   # Spin-down

        # Initialize the BdG Hamiltonian
        if isComplex
            HBdG = zeros(Complex{Float64}, 2 * nSites, 2 * nSites)
        else
            HBdG = zeros(Float64, 2 * nSites, 2 * nSites)
        end

        for i in 1:nSites
            if i == impuritySite
                H[i, i] = -(μ - (U * n_up[i]) - (K - σ_dn * J))
            else
                H[i, i] = -(μ - (U * n_up[i]))
            end
        end

        for i in 1:nSites
            for j in 1:nSites
                HBdG[i, j] = H[i, j]
                HBdG[nSites + i, nSites + j] = -conj(H[i, j])
            end
        end

        # Add pairing terms D_ij in the off-diagonal
        for i in 1:nSites
            HBdG[i, nSites + i] = deltaList[i]          # D_ij in upper-left block
            HBdG[nSites + i, i] = conj(deltaList[i])    # D_ij^* in the upper-left block
        end

        for i in 1:nSites
            if i == impuritySite
                HBdG[nSites + i, nSites + i] = (μ - (U * n_dn[i]) - (K - σ_up * J))
            else
                HBdG[nSites + i, nSites + i] = (μ - (U * n_dn[i]))
            end
        end
        return HBdG
    end


    function run_self_consistency_numpy(deltaOld, μ, nSites, n_up, n_dn, tMat, U, J, impuritySite, T; tol=1e-5, maxCount=150, isComplexCalc=false)
        startTime = time()
        count = 0
        flag = false

        while flag == false
            count += 1
            H1 = create_bdg_ham_up(deltaOld, tMat, μ, nSites, n_up, n_dn, U, impuritySite, J, isComplex=isComplexCalc)
            H2 = create_bdg_ham_dn(deltaOld, tMat, μ, nSites, n_up, n_dn, U, impuritySite, J, isComplex=isComplexCalc)

            if !is_symmetric(H1) || !is_symmetric(H2)
                println("Error: Matrix not symmetric")
            end
            (evals1, evecs1) = np.linalg.eigh(H1)
            un_up, vn_dn = sort_evecs(evecs1, nSites)

            (evals2, evecs2) = np.linalg.eigh(H2)
            un_dn, vn_up = sort_evecs(evecs2, nSites)

            evalPostive1 = evals1[nSites + 1 : end]
            evalPostive2 = evals2[nSites + 1 : end]

            deltaNew = calc_delta(U, un_up, vn_up, un_dn, vn_dn, nSites, evalPostive1, evalPostive2, T)
            n_up_new = calc_avg_n(un_up, vn_dn, nSites, evalPostive1, evalPostive2, T)
            n_dn_new = calc_avg_n(un_dn, vn_up, nSites, evalPostive1, evalPostive2, T)

            deltaFlag = check_rel_tol(deltaOld, deltaNew, tol=tol)
            if deltaFlag == true
                deltaFinal = deltaNew
                n_up_final = n_up_new
                n_dn_final = n_dn_new
                nAvg = n_up_final + n_dn_final
                isConverged = true
                endTime = round(time() - startTime, digits = 2)
                return deltaFinal, n_up_final, n_dn_final, nAvg, evecs1, evecs2, evals1, evals2, isConverged, endTime, count

                break
            else
                deltaOld = deltaNew
                if count >= maxCount
                    deltaFinal = deltaNew
                    n_up_final = n_up_new
                    n_dn_final = n_dn_new
                    nAvg = n_up_final + n_dn_final
                    endTime = round(time() - startTime, digits = 2)
                    isConverged = false
                    flag = true
                    return deltaFinal, n_up_final, n_dn_final, nAvg, evecs1, evecs2, evals1, evals2, isConverged, endTime, count
                    break
                end
            end
        end
    end

end # Module end

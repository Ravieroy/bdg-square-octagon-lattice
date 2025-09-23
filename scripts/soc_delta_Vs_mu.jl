#-----------------------------------
#   DESCRIPTION: This program calculates the order parameter(Δ) for various
#   values of chemical potential μ for a spinfull case. λ ≡ Rashba SOC
#   λ_iso ≡ Intrinsic SOC

using Distributed
@everywhere begin
    using DelimitedFiles
    using LinearAlgebra
    using Dates
    using PyCall
    using Statistics
    using CSV
    using DataFrames
    @everywhere(@pyimport numpy as np)
    # Importing personal modules
    include("../src/bdg_utilities.jl")
    include("../src/general_utils.jl")
    include("../src/logging_utils.jl")
    include("params.jl")
end # module everywhere end

# Loading personal modules
@everywhere using .BdGUtilities
@everywhere using .GeneralUtils
@everywhere using .LoggingUtils

@everywhere function main(μ)
    st = time()
    logFileName = "$(logFolder)soc_mu_$μ.log"
    fileName = "$(saveInFolder)soc_mu_$μ.txt"
    deltaFinal, _, _, nAvg, _, _, isConverged, endTime, count =
        run_self_consistency_numpy_spinfull(deltaOld,
            μ,
            nSites,
            n_up,
            n_dn,
            tMat,
            U,
            J,
            x, y,
            neighbors,
            λ, λ_iso,
            t_map,
            iso_map,
            impuritySite,
            T,
            tol=tol,
            maxCount=maxCount,
            isComplexCalc=isComplexCalc)

    if isConverged == true
        message = string("Converged in $count iterations in $endTime s")
        write_log(logFileName, "SUCCESS", message)
    else
        message = string("Calculation did not converge. Total time: $endTime s")
        write_log(logFileName, "WARNING", message)
    end

    Δ̄ = mean(abs.(deltaFinal))
    n̄ = mean(abs.(nAvg))
    list = [μ, Δ̄, n̄]
    open(fileName, "a") do io
        println(io, join(list, '\t'))
    end
    deltaFinal = nothing
    Δ̄ = nothing
    n̄ = nothing
    et = time() - st
    message = string("Completed in: ", format_elapsed_time(et))
    write_log(logFileName, "INFO", message)
    GC.gc()
end

timestart = time()
@everywhere begin
    # tMatfileName = "$(dataSetFolder)ham_$(N)_t_$t"
    tMat = generate_t_matrix(tMatfileName)
    df = CSV.read("$(dataSetFolder)df_square-octagon$N.csv", DataFrame)
    x = df.x
    y = df.y
    neighbors = hcat(df.n1, df.n2, df.n3, df.n4)
    t_map = generate_t_map(df, t2)
    iso_map = compute_iso_map(df)
    # iso_map = nothing
end
infoLogFileName = "$(logFolder)info.log"
pf = @__FILE__

# Detailed description of the program
desc = """
Calculates the order parameter (Δ) vs. μ for the spinfull case.
The program takes μVals as input and processes each value in parallel.
For each μ, the program calculates and stores the following data:
    - μ: Chemical Potential
    - Δ: Order parameter
"""

if !isfile(infoLogFileName)
    message = string("Program name $pf")
    write_log(infoLogFileName, "INFO", message)
    write_log(infoLogFileName, desc)
end

fileName = "$(saveInFolder)soc_mu.txt"
if !isfile(fileName)
    list = ["μ", "Δ", "nAvg"]
    open(fileName, "a") do io
        println(io, join(list, '\t'))
    end
end

pmap(main, μVals)
elapsed = time() - timestart
timeNow, dateToday = get_present_time()
println("($timeNow)Total Time Taken: ", format_elapsed_time(elapsed))

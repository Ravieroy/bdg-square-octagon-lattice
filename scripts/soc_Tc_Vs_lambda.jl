#-----------------------------------
#   DESCRIPTION: This program calculates critical temperature Tc Vs λ
#   for the spinfull case. λ ≡ Rashba SOC
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

@everywhere function main(λ)
    logFileName = "$(logFolder)soc_lambda_$λ.log"
    fileName = "$(saveInFolder)soc_Tc_lambda_$λ.txt"
    message = string("Running for λ=$λ, id=$(myid())")
    write_log(logFileName, message)
    st = time()
    dT = 0.001
    # T = 0.15
    T = T_start_map[λ]
    foundTc = false
    iter = 1
    while foundTc == false
        message = string("Running for T=$T")
        write_log(logFileName, "INFO", message)
        n_up = 0 * ones(Float64, nSites)
        n_dn = 0 * ones(Float64, nSites)
        deltaOld = ones(ComplexF64, nSites)
        deltaFinal, _, _, _, _, _, isConverged, endTime, count =
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
            message = string("Calculation did not converge in $endTime s")
            write_log(logFileName, "WARNING", message)
        end

        Δ̄ = mean(abs.(deltaFinal))
        message = string("iter=$iter, Δ̄=$Δ̄")
        write_log(logFileName, message)

        if Δ̄ < 0.01
            foundTc = true
            Δ̄ = round(Δ̄, digits=3)
            Tc = round(T, digits=3)
            list = [λ, Tc]
            open(fileName, "a") do io
                println(io, join(list, '\t'))
            end
        end
        iter += 1
        T = T + dT
    end # while(foundTc) loop end

    et = round(time() - st, digits=2)
    column_names = ["λ", "Tc"]
    tcFileNameFinal = "$(saveInFolder)Tc.txt"
    # Open the file for writing
    if !isfile(tcFileNameFinal)
        open(tcFileNameFinal, "w") do io
            # Write column names as the first row
            println(io, join(column_names, "\t"))
        end
    end
    message = string("Completed in $et seconds")
    write_log(logFileName, "INFO", message)
    GC.gc()
end  # main end

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

    T_start_map = Dict(
    0.0 => 0.2,
    0.2 => 0.17,
    )
end

infoLogFileName = "$(logFolder)info.log"
pf = @__FILE__

# Detailed description of the program
desc = """
Calculates the critical temperature Tc for various chemical potential λ.
for the spinfull case.
The program takes μVals as input and processes each value in parallel.
For each μ, the program calculates and stores the following data:
    - λ : Chemical Potential
    - Tc : critical temperature
"""

if !isfile(infoLogFileName)
    message = string("Program name $pf")
    write_log(infoLogFileName, "INFO", message)
    write_log(infoLogFileName, desc)
end

pmap(main, λVals)
elapsed = round(time() - timestart, digits=2)
timeNow, dateToday = get_present_time()
println("($timeNow)The elapsed time : $elapsed secs ($(round(elapsed/60, digits = 2)) mins)")


#-----------------------------------
#   DESCRIPTION: This program calculates the order parameter(Δ) for various
#   values of on-site interaction U and λ for spinfull case.
#
#   It runs the program for given values of λ in parallel and stores the
#   results in a text file. This program runs the U-loop in series and thus
#   it will be slower if only one calculation for λ is needed. It is better to
#   run for U in parallel and get the results much faster.
#
#   This program can be used in conjunction with makegrid.py to generate a 3D
#   plot in the U-μ-Δ grid.

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
    logFileName = "$(logFolder)lambda_$λ.log"
    fileName = "$(saveInFolder)lambda_$λ.txt"
    list = ["T", "ΔAvg"]
    message = string("Running for λ=$λ, id=$(myid())")
    write_log(logFileName, message)
    if !isfile(fileName)
        open(fileName, "a") do io
            println(io, join(list, '\t'))
        end
    end

    for T in TVals
        st = time()
        message = string("Running for T=$T")
        write_log(logFileName, message)
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
        list = [T, Δ̄]
        open(fileName, "a") do io
            println(io, join(list, '\t'))
        end
        deltaFinal = nothing
        Δ̄ = nothing
        et = time() - st
        message = string("Completed for T=$T in ", format_elapsed_time(et))
        write_log(logFileName, message)
    end
    GC.gc()
end  # main end

timestart = time()
@everywhere begin
    tMat = generate_t_matrix(tMatfileName)
    df = CSV.read("$(dataSetFolder)df_square-octagon$N.csv", DataFrame)
    x = df.x
    y = df.y
    neighbors = hcat(df.n1, df.n2, df.n3, df.n4)
    t_map = generate_t_map(df, t2)
    # iso_map = compute_iso_map(df)
    iso_map = nothing
end

infoLogFileName = "$(logFolder)info.log"
pf = @__FILE__

# Detailed description of the program
desc = """
Calculates the order parameter (Δ) vs. Temperature (T).
The program takes λVals as input and processes each value in parallel.
For each λ, the program calculates and stores the following data:
    - T: Temperature
    - ΔAvg: Average order parameter(Δ=ΔAvg for J=0)
"""

if !isfile(infoLogFileName)
    message = string("Program name $pf")
    write_log(infoLogFileName, "INFO", message)
    write_log(infoLogFileName, desc)
end

pmap(main, λVals)
elapsed = time() - timestart
timeNow, dateToday = get_present_time()
println("($timeNow)Total Time Taken: ", format_elapsed_time(elapsed))


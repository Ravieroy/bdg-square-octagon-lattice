#-----------------------------------
#   DESCRIPTION: This program calculates the order parameter(Δ)
#   Vs T for various values of chemical potential μ.
#
#   It runs the program for given values of μ in parallel and stores the
#   results in a text file.

using Distributed
@everywhere begin
    using DelimitedFiles
    using LinearAlgebra
    using Dates
    using PyCall
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
    logFileName = "$(logFolder)mu_$μ.log"
    fileName = "$(saveInFolder)mu_$μ.txt"
    message = string("Running for mu=$μ, id=$(myid())")
    write_log(logFileName, message)
    list = ["T", "ΔAvg"]
    if !isfile(fileName)
        open(fileName, "a") do io
            println(io, join(list, '\t'))
        end
    end

    for T in TVals
        st = time()
        message = string("Running for T=$T")
        write_log(logFileName, message)
        deltaFinal, _, _, nAvgFinal, _, _, _, _, isConverged, endTime, count =
            run_self_consistency_numpy(
                deltaOld,
                μ,
                nSites,
                n_up,
                n_dn,
                tMat,
                U,
                J,
                impuritySite,
                T,
                tol=tol,
                maxCount=maxCount,
                isComplexCalc=isComplexCalc
            )

        if isConverged == true
            message = string("Converged in $count iterations")
            write_log(logFileName, "SUCCESS", message)
            Δ̄ = sum(deltaFinal) / nSites
        else
            message = string("Calculation did not converge")
            write_log(logFileName, "WARNING", message)
            Δ̄ = sum(deltaFinal) / nSites
            # Δ̄ = NaN  # Use NaN for non-converging points
        end

        if !isreal(Δ̄)
            Δ̄ = abs(Δ̄)
        end
        list = [T, Δ̄]

        open(fileName, "a") do io
            println(io, join(list, '\t'))
        end

        deltaFinal = nothing
        nAvgFinal = nothing
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
    if !isreal(tMat)
        tMat = real(tMat)
    end
end

infoLogFileName = "$(logFolder)info.log"
pf = @__FILE__

# Detailed description of the program
desc = """
Calculates the order parameter (Δ) vs. temperature (T).
The program takes μVals as input and processes each value in parallel.
For each μ, the program calculates and stores the following data:
    - T: Temperature
    - ΔAvg: Average order parameter(Δ=ΔAvg for J=0)
"""
if !isfile(infoLogFileName)
    message = string("Program name $pf")
    write_log(infoLogFileName, "INFO", message)
    write_log(infoLogFileName, desc)
end

pmap(main, μVals)
elapsed = time() - timestart
timeNow, dateToday = get_present_time()
println("($timeNow)Total Time Taken: ", format_elapsed_time(elapsed))

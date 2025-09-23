#-----------------------------------
#   DESCRIPTION: This program calculates the order parameter(Δ) for various
#   values of chemical potential μ.
#
#   It runs the program for given values of μ in parallel and stores the
#   results in a text file.
#
#   See paper PRB. 105, 214510 (2022) by Agnieszka Cichy et.al(Fig. 2c)
#
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
    st = time()
    logFileName = "$(logFolder)mu_$μ.log"
    fileName = "$(saveInFolder)mu_$μ.txt"

    deltaFinal, _, _, _, _, _, _, _, isConverged, endTime, count =
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
    else
        message = string("Calculation did not converge")
        write_log(logFileName, "WARNING", message)
    end

    Δ̄ = sum(deltaFinal) / nSites
    if !isreal(Δ̄)
        Δ̄ = abs(Δ̄)
    end
    list = [μ, Δ̄]
    open(fileName, "a") do io
        println(io, join(list, '\t'))
    end
    deltaFinal = nothing
    Δ̄ = nothing
    et = time() - st
    message = string("Completed in: ", format_elapsed_time(et))
    write_log(logFileName, "INFO", message)
    GC.gc()
end

timestart = time()
@everywhere begin
    # tMatfileName = "$(dataSetFolder)ham_$(N)_t_$t"
    tMat = generate_t_matrix(tMatfileName)
    if !isreal(tMat)
        tMat = real(tMat)
    end
end
infoLogFileName = "$(logFolder)info.log"
pf = @__FILE__

# Detailed description of the program
desc = """
Calculates the order parameter (Δ) vs. μ.
The program takes μVals as input and processes each value in parallel.
For each μ, the program calculates and stores the following data:
    - μ: Chemical Potential
    - ΔAvg: Average order parameter(Δ=ΔAvg for J=0)
"""

if !isfile(infoLogFileName)
    message = string("Program name $pf")
    write_log(infoLogFileName, "INFO", message)
    write_log(infoLogFileName, desc)
end

fileName = "$(saveInFolder)mu.txt"
if !isfile(fileName)
    list = ["μ", "ΔAvg"]
    open(fileName, "a") do io
        println(io, join(list, '\t'))
    end
end

pmap(main, μVals)
elapsed = time() - timestart
timeNow, dateToday = get_present_time()
println("($timeNow)Total Time Taken: ", format_elapsed_time(elapsed))

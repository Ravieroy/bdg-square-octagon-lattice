#-----------------------------------
#   DESCRIPTION: This program calculates the order parameter(Δ) for various
#   values of t'/t.
#
#   It runs the program for given values of t'/t in parallel and stores the
#   results in a text file.
#
#   See paper PRB. 105, 214510 (2022) by Agnieszka Cichy et.al(Fig. 2d)
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

@everywhere function main(t2)
    st = time()
    logFileName = "$(logFolder)t2_$t2.log"
    fileName = "$(saveInFolder)t2_$t2.txt"
    tMatfileName = "$(dataSetFolder)ham_$(N)_t2_$t2"
    tMat = generate_t_matrix(tMatfileName)
    if !isreal(tMat)
        tMat = real(tMat)
    end

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
    list = [t2, Δ̄]
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

infoLogFileName = "$(logFolder)info.log"
pf = @__FILE__

# Detailed description of the program
desc = """
Calculates the order parameter (Δ) vs. t'/t.
The program takes t2Vals as input and processes each value in parallel.
For each t, the program calculates and stores the following data:
    - t'/t: ratio of NNN and NN hopping parameter.
    - ΔAvg: Average order parameter(Δ=ΔAvg for J=0)
"""

if !isfile(infoLogFileName)
    message = string("Program name $pf")
    write_log(infoLogFileName, "INFO", message)
    write_log(infoLogFileName, desc)
end

fileName = "$(saveInFolder)t2.txt"
if !isfile(fileName)
    list = ["t2", "ΔAvg"]
    open(fileName, "a") do io
        println(io, join(list, '\t'))
    end
end

pmap(main, t2Vals)
elapsed = time() - timestart
timeNow, dateToday = get_present_time()
println("($timeNow)Total Time Taken: ", format_elapsed_time(elapsed))

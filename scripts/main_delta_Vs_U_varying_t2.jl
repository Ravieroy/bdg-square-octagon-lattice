#-----------------------------------
#   DESCRIPTION: This program calculates the order parameter(Δ) for various
#   values of on-site interaction U and ratio of NNN and NN hopping parameter
#   t2/t1.
#
#   It runs the program for given values of t2/1 in parallel and stores the
#   results in a text file. This program runs the U-loop in series and thus
#   it will be slower if only one calculation for t' is needed. It is better to
#   run for U in parallel and get the results much faster.
#
#   See paper PRB. 105, 214510 (2022) by Agnieszka Cichy et.al(Fig. 1)
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
    logFile = "$(logFolder)t2_$t2.log"
    fileName = "$(saveInFolder)t2_$t2.txt"
    list = ["U", "deltaAvg"]
    open(fileName, "a") do io
        println(io, join(list, '\t'))
    end
    tMatfileName = "$(dataSetFolder)ham_$(N)_t2_$t2"
    tMat = generate_t_matrix(tMatfileName)
    if !isreal(tMat)
        tMat = real(tMat)
    end
    for U in UVals
        st = time()
        message = string("Running for t2=$t2, U=$U")
        write_log(logFile, message)

        # initial guesses
        nUp = copy(nUp0)
        nDn = copy(nDn0)
        deltaOld = ones(Float64, nSites)

        deltaFinal, _, _, nAvg, _, _, _, _, isConverged, endTime, count =
            run_self_consistency_numpy(
                deltaOld,
                mu, nSites,
                nUp, nDn,
                tMat, U, J,
                impuritySite, T;
                tol=tol, maxCount=maxCount,
                includeHartree=includeHartree,
                verboseLogIn=logFile
            )
        if isConverged == true
            message = string("Converged in $count iterations")
            write_log(logFile, "SUCCESS", message)
            deltaAvg = sum(deltaFinal) / nSites
        else
            message = string("Calculation did not converge")
            write_log(logFile, "WARNING", message)
            deltaAvg = sum(deltaFinal) / nSites
        end

        if !isreal(deltaAvg)
            deltaAvg = abs(deltaAvg)
        end
        list = [U, deltaAvg]
        open(fileName, "a") do io
            println(io, join(list, '\t'))
        end
        deltaFinal = nothing
        deltaAvg = nothing

        et = time() - st
        message = string("Completed for t2=$t2, U=$U in ", format_elapsed_time(et))
        write_log(logFile, message)
    end
    GC.gc()
end  # main end

timestart = time()

infoLogFileName = "$(logFolder)info.log"
pf = @__FILE__

# Detailed description of the program
desc = """
Calculates the order parameter (delta) vs. interaction strength (U).
The program takes t2Vals as input and processes each value in parallel.
For each t, the program calculates and stores the following data:
    - U: interaction strength
    - deltaAvg: Average order parameter(Δ=deltaAvg for J=0)
"""

if !isfile(infoLogFileName)
    message = string("Program name $pf")
    write_log(infoLogFileName, "INFO", message)
    write_log(infoLogFileName, desc)
end

pmap(main, t2Vals)
elapsed = time() - timestart
timeNow, dateToday = get_present_time()
println("($timeNow)Total Time Taken: ", format_elapsed_time(elapsed))


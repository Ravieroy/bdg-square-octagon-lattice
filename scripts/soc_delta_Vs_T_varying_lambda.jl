#-----------------------------------
#   DESCRIPTION: This program calculates the order parameter(Δ) for various
#   values of temperature T and lambda for spinfull case.
#
#   It runs the program for given values of lambda in parallel and stores the
#   results in a text file. This program runs the T-loop in series and thus
#   it will be slower if only one calculation for lambda is needed. It is better
#   to run for T in parallel and get the results much faster.
#
#   Following common error means that check_rel_tol got two different types of
#   inputs. This usually happens if the calculation will have Complex values
#   and initial values provided for deltaOld is Float. So this particular script
#   in its present form won't work for lambda=0. You can make it work with a
#   conditional statement, or simply use the non-spinfull version of this
#
#   ERROR: LoadError: MethodError: no method matching
#   The function `check_rel_tol` exists, but no method is defined for
#   this combination of argument types.

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

@everywhere function main(lambda)
    logFile = "$(logFolder)lambda_$lambda.log"
    fileName = "$(saveInFolder)lambda_$lambda.txt"
    list = ["T", "deltaAvg"]
    message = string("Running for lambda=$lambda, id=$(myid())")
    write_log(logFile, message)
    if !isfile(fileName)
        open(fileName, "a") do io
            println(io, join(list, '\t'))
        end
    end

    for T in TVals
        st = time()
        message = string("Running for T=$T")
        write_log(logFile, message)

        # initial guesses
        nUp = copy(nUp0)
        nDn = copy(nDn0)
        deltaOld = copy(deltaOld0)

        deltaFinal, _, _, nAvg, evecs, evals, isConverged, endTime, count =
            run_self_consistency_numpy_spinfull(
            deltaOld,
            mu, nSites,
            nUp, nDn,
            tMat, U, J,
            x, y, neighbors,
            lambda, lambdaIso,
            tMap, isoMap,
            impuritySite, T;
            tol=tol, maxCount=maxCount,
            includeHartree=includeHartree,
            verboseLogIn=logFile
        )

        if isConverged == true
            message = string("Converged in $count iterations in $endTime s")
            write_log(logFile, "SUCCESS", message)
        else
            message = string("Calculation did not converge in $endTime s")
            write_log(logFile, "WARNING", message)
        end

        deltaAvg = mean(abs.(deltaFinal))
        list = [T, deltaAvg]
        open(fileName, "a") do io
            println(io, join(list, '\t'))
        end
        deltaFinal = nothing
        deltaAvg = nothing
        et = time() - st
        message = string("Completed for T=$T in ", format_elapsed_time(et))
        write_log(logFile, message)
    end
    GC.gc()
end  # main end

timestart = time()
@everywhere begin
    df = CSV.read(rawdfName, DataFrame)
    x, y, neighbors = extract_geometry(df)
    tMap = generate_t_map(neighbors; t1=1.0, t2=t2)
    tMat = generate_t_matrix(tMatFileName)
    # iso_map = compute_iso_map(df)
    iso_map = nothing
end

infoLogFileName = "$(logFolder)info.log"
pf = @__FILE__

# Detailed description of the program
desc = """
Calculates the order parameter (Δ) vs. Temperature (T).
The program takes lambdaVals as input and processes each value in parallel.
For each lambda, the program calculates and stores the following data:
    - T: Temperature
    - deltaAvg: Average order parameter(Δ=deltaAvg)
"""

if !isfile(infoLogFileName)
    message = string("Program name $pf")
    write_log(infoLogFileName, "INFO", message)
    write_log(infoLogFileName, desc)
end

pmap(main, lambdaVals)
elapsed = time() - timestart
timeNow, dateToday = get_present_time()
println("($timeNow)Total Time Taken: ", format_elapsed_time(elapsed))


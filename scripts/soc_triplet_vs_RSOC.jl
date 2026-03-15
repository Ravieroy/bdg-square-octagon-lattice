using Distributed

@everywhere begin
    using DelimitedFiles
    using LinearAlgebra
    using Statistics
    using Dates
    using CSV
    using DataFrames
    using PyCall
    using Statistics
    @everywhere(@pyimport numpy as np)

    include("../src/bdg_utilities.jl")
    include("../src/general_utils.jl")
    include("../src/logging_utils.jl")
    include("params.jl")
end

@everywhere using .BdGUtilities
@everywhere using .GeneralUtils
@everywhere using .LoggingUtils


@everywhere begin
    function init_shared(rawdfName::String, tMatFileName::String, t2::Float64)
        df = CSV.read(rawdfName, DataFrame)
        x, y, neighbors = extract_geometry(df)
        tMap = generate_t_map(neighbors; t1=1.0, t2=t2)
        tMat = generate_t_matrix(tMatFileName)
        return x, y, neighbors, tMap, tMat
    end

    const x, y, neighbors, tMap, tMat = init_shared(rawdfName, tMatFileName, t2)
end

@everywhere function main(lambda::Float64)
    logFile = "$(logFolder)soc_triplet_lambda_$(lambda)_pid_$(myid()).log"
    outFile = "$(saveInFolder)spin_triplet_lambda_$(lambda).txt"

    st = time()
    write_log(logFile, "INFO", "Running lambda=$lambda on worker $(myid())")

    nUp = copy(nUp0)
    nDn = copy(nDn0)

    if lambda == 0.0
        deltaOld = ones(Float64, nSites)
        isComplexCalc = false
    else
        deltaOld = ones(ComplexF64, nSites)
        isComplexCalc = true
    end

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
            verboseLogIn=logFile
        )

    if isConverged
        write_log(logFile, "SUCCESS", "Converged in $count iterations; $endTime s")
    else
        write_log(logFile, "WARNING", "Not converged in $count iterations; $endTime s")
    end

    un_up, vn_dn, un_dn, vn_up, Epos =
        sort_evecs_spinfull(evecs, evals, nSites; tol=0.0)

    F = compute_pairing_correlators(
        un_up, vn_up,
        un_dn, vn_dn,
        Epos, T, nSites
    )

    diag = triplet_diagnostics(F, neighbors)
    triplet_total = sqrt(diag.rms_eqspin^2 + diag.rms_mz0^2)
    mean_abs_delta = mean(abs.(deltaFinal))
    ratio_ts   = triplet_total / (mean_abs_delta + eps())

    open(outFile, "a") do io
        println(io, join([
                lambda,
                triplet_total,
                diag.rms_eqspin,
                diag.rms_mz0,
                mean_abs_delta,
                ratio_ts
            ], '\t'))
    end

    et = round(time() - st, digits=2)
    write_log(logFile, "INFO", "Completed lambda=$lambda in $(format_elapsed_time(et))")

    F = nothing
    un_up = nothing
    vn_dn = nothing
    un_dn = nothing
    vn_up = nothing
    Epos = nothing
    evecs = nothing
    evals = nothing
    deltaFinal = nothing
    nUp = nothing
    nDn = nothing
    nAvg = nothing
    diag = nothing
    deltaOld = nothing

    pyimport("gc")[:collect]()
    GC.gc()

    return nothing
end

timestart = time()

# Detailed description of the program
desc = """
Calculates the spin triplet induced due to Rashba SOC.
The program takes lambdaVals as input and processes each value in parallel.
For each lambda, the program calculates and stores the following data:
    - lambda : Rashba SOC strength
    - triplet_total : Total contribution of triplet order parameter (rms_eqspin+rms_mz0)
    - rms_eqspin : Contribution from equal spin (↑↑ and ↓↓)
    - rms_mz0 : Contribution from mz = 0 case
    - mean_abs_delta : Superconducting order parameter (Δ)
    - ratio_ts : Ratio between triplet and singlet contribution
"""

infoLogFileName = "$(logFolder)info.log"
pf = @__FILE__
if !isfile(infoLogFileName)
    message = string("Program name $pf")
    write_log(infoLogFileName, "INFO", message)
    write_log(infoLogFileName, desc)
end


fileName = "$(saveInFolder)spin_triplet.txt"
if !isfile(fileName)
    open(fileName, "w") do io
        println(io, join([
                "lambda",
                "triplet_total",
                "rms_eqspin",
                "rms_mz0",
                "mean_abs_delta",
                "ratio_ts",
            ], '\t'))
    end
end

pmap(main, lambdaVals)

elapsed = round(time() - timestart, digits=2)
timeNow, dateToday = get_present_time()
println("($timeNow)Total Time Taken: ", format_elapsed_time(elapsed))

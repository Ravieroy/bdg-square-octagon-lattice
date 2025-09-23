# -------------------------------
#   This program calculates the order parameter(Δ) Vs μ.
#   See paper arXiv:2403.06231v1 by S.Ding et.al(Fig. 1)
#   J. Phys. Chem. Lett. 2024, 15, 9084−9091(Fig. 1b, 1e, 1f)
#
#
#   It takes μVals as variable to be run in multiple CPUs
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
    logFileName = "$(logFolder)mu_$μ.log"
    fileName = "$(saveInFolder)mu_$μ.txt"
    message = string("Running for mu=$μ ,Θ=$Θ, J=$J")
    write_log(logFileName, message)
    deltaFinal, _, _, nAvgFinal, _, _, _, _, isConverged, endTime, count =
        run_self_consistency_numpy(deltaOld,
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
            isComplexCalc=isComplexCalc)


    if isConverged == true
        message = string("Converged in $count iterations")
        write_log(logFileName, "SUCCESS", message)
    else
        message = string("Calculation did not converge")
        write_log(logFileName, "WARNING", message)
    end

    if !isreal(deltaFinal)
      deltaFinal = [abs(z) for z in deltaFinal]
    end

    ΔAvg = sum(deltaFinal) / nSites
    nAvg = sum(nAvgFinal) / nSites
    list = [μ, ΔAvg, nAvg]
    open(fileName, "a") do io
        println(io, join(list, '\t'))
    end
    message = string("Completed for mu=$μ")
    write_log(logFileName, message)
end  # main end

timestart = time()
@everywhere begin
    tMat = generate_t_matrix(tMatfileName)
    if isreal(tMat)
        tMat = real(tMat)
    end
end

pmap(main, μVals)
elapsed = round(time() - timestart, digits=2)
timeNow, dateToday = get_present_time()
println("($timeNow)The elapsed time : $elapsed secs ($(round(elapsed/60, digits = 2)) mins)")
fname = "../results/delta_Vs_mu.csv"
headers = ["μ", "ΔAvg", "nAvg"]
if !isfile(fname)
    open(fname, "a") do io
        println(io, join(headers, '\t'))
    end
end

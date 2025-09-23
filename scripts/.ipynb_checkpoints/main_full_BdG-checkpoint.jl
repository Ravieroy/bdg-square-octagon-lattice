# Importing required libraries
using Distributed
@everywhere begin
    using LatticeUtilities
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
    nTotalJobs = length(ΘVals) * length(JVals) * length(tempList)
    nJob = 0

    deltaDict = Dict()
    nAvgDict = Dict()
    deltaImpDict = Dict()
    eShibaDict = Dict()
    evecs1Dict = Dict()
    evecs2Dict = Dict()
    evalsDict = Dict()
    for Θ in ΘVals
        deltaDictJ = Dict()
        nAvgDictJ = Dict()
        deltaImpDictJ = Dict()
        eShibaDictJ = Dict()
        evecs1DictJ = Dict()
        evecs2DictJ = Dict()
        evalsDictJ = Dict()

        tMatfileName = "../data/ham_$(N)_theta_$(Θ)"
        tMat = generate_t_matrix(tMatfileName)
        if isreal(tMat)
            tMat = real(tMat)
        end
        if Θ == 0 || Θ == 1
            deltaOld = ones(Float64, nSites)
            isComplexCalc = false
        else
            deltaOld = ones(ComplexF64, nSites)
            isComplexCalc = true
        end

        for J in JVals
            deltaDictT = Dict()
            nAvgDictT = Dict()
            deltaImpDictT = Dict()
            eShibaDictT = Dict()
            evecs1DictT = Dict()
            evecs2DictT = Dict()
            evalsDictT = Dict()
            for T in tempList
                nJob += 1
                #------Logs-------
                message = string("Running for Θ=$Θ, J=$J, T=$T")
                write_log(logFileName, message)
                deltaFinal, _, _, nAvgFinal, evecs1, evecs2, evals1, evals2, isConverged, endTime =
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
                    message = string("Converged for Θ=$Θ, J=$J, T=$T")
                    write_log(logFileName, "SUCCESS", message)
                else
                    message = string("HitMaxCount for Θ=$Θ, J=$J, T=$T")
                    write_log(logFileName, "WARNING", message)
                end

                evals = evals1
                E_YSR = evals[nSites+1:end][1]

                if !isreal(deltaFinal)
                    deltaFinal = [abs(z) for z in deltaFinal]
                end

                ΔImp = deltaFinal[impuritySite]

                deltaDictT[T] = deltaFinal
                nAvgDictT[T] = nAvgFinal
                deltaImpDictT[T] = ΔImp
                eShibaDictT[T] = E_YSR
                evecs1DictT[T] = evecs1
                evecs2DictT[T] = evecs2
                evalsDictT[T] = evals
            end # T loop end
            deltaDictJ[J] = deltaDictT
            nAvgDictJ[J] = nAvgDictT
            deltaImpDictJ[J] = deltaImpDictT
            eShibaDictJ[J] = eShibaDictT
            evecs1DictJ[J] = evecs1DictT
            evecs2DictJ[J] = evecs2DictT
            evalsDictJ[J] = evalsDictT
        end #V loop end
        deltaDict[Θ] = deltaDictJ
        nAvgDict[Θ] = nAvgDictJ
        deltaImpDict[Θ] = deltaImpDictJ
        eShibaDict[Θ] = eShibaDictJ
        evecs1Dict[Θ] = evecs1DictJ
        evecs2Dict[Θ] = evecs2DictJ
        evalsDict[Θ] = evalsDictJ
    end # alpha loop end
    # code block to save dictionary locally
    dictList = [deltaDict, nAvgDict, deltaImpDict, eShibaDict, evecs1Dict, evecs2Dict, evalsDict]
    for (key, value) in store
        if value[1] == true
            baseFileName = value[2]
            newFileName = string(saveInFolder,
                baseFileName,
                "_",
                μ,
                ".", fileFormat
            )
            save_file(dictList[value[3]], newFileName)
        end
    end # store loop end
    et = round(time() - st, digits=2)
    message = string("Completed in $et mins")
    write_log(logFileName, message)
    # free up memory
    deltaDict = nothing
    nAvgDict = nothing
    deltaImpDict = nothing
    eShibaDict = nothing
    evecsDict = nothing
    evalsDict = nothing

    deltaDictT = nothing
    nAvgDictT = nothing
    deltaImpDictT = nothing
    eShibaDictT = nothing
    evecsDictT = nothing
    evalsDictT = nothing

    deltaDictV = nothing
    nAvgDictV = nothing
    deltaImpDictV = nothing
    eShibaDictV = nothing
    evecsDictV = nothing
    evalsDictV = nothing
    GC.gc()
end  # main end

timestart = time()
pmap(main, μVals)
elapsed = round(time() - timestart, digits=2)
timeNow, dateToday = get_present_time()
roundedElapsed = round(elapsed / 60, digits=2)
println("($timeNow)The elapsed time : $elapsed secs ($roundedElapsed mins)")

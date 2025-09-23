#-----------------------------------
#   DESCRIPTION: This program calculates critical temperature Tc Vs t₂ with
#   Rashba SOC allowed
#
using Distributed
@everywhere begin
    using DelimitedFiles
    using LinearAlgebra
    using Dates
    using PyCall
    using Statistics
    using CSV
    using DataFrames
    using Printf
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
    fileName = "$(saveInFolder)Tc_t2_$t2.txt"
    message = string("Running for t2=$t2, id=$(myid())")
    write_log(logFileName, message)

    tMatfileName = "$(dataSetFolder)ham_$(N)_t2_$t2"
    tMat = generate_t_matrix(tMatfileName)
    t_map = generate_t_map(df, t2) # we initialize it here because Rashba SOC depends on t2

    # === CSV logging setup ===
    csvfile = "$(logFolder)Tc_calculation_log_$t2.csv"
    open(csvfile, "w") do io
        println(io, "iter,timestamp,T_high,T_low,T_try,converged,Δ̄,T_diff")
    end
    # =========================
    T_low = 0.0
    T_high = 0.5
    T_tol = 5e-3
    max_iter = 40
    gap_threshold = 1e-2

    # helper to evaluate gap at a temperature; returns (ok, Δ̄)
    function eval_gap_at(T_try)
        n_up = zeros(Float64, nSites)
        n_dn = zeros(Float64, nSites)
        deltaOld = ones(ComplexF64, nSites)   # use Complex for consistency with other code
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
                T_try,
                tol=tol,
                maxCount=maxCount,
                isComplexCalc=isComplexCalc)
        if !isConverged
            return (false, NaN)
        end
        return (true, mean(abs.(deltaFinal)))
    end

    # Ensure bracket: f(T) = Δ̄(T) - gap_threshold should satisfy f(low) > 0 and f(high) < 0
    ok_low, Δ_low = eval_gap_at(T_low)
    ok_high, Δ_high = eval_gap_at(T_high)

    if !ok_low
        write_log(logFileName, "WARNING", "Evaluation at T_low=$T_low did not converge; trying small positive T_low")
        ok_low, Δ_low = eval_gap_at(1e-4)
        if !ok_low
            write_log(logFileName, "ERROR", "Couldn't evaluate Δ at low T. Storing Tc = $T_low")
            open(fileName, "a") do io
                println(io, "$t2\t$T_low")
            end
            return
        else
            T_low = 1e-4
        end
    end

    if Δ_low <= gap_threshold
        # No superconductivity even at lowest T — Tc = 0 (or T_low)
        write_log(logFileName, "INFO", "Gap at T_low=$T_low already ≤ threshold. Setting Tc = $T_low")
        open(fileName, "a") do io
            println(io, "$t2\t$T_low")
        end
        return
    end

    if !ok_high
        write_log(logFileName, "WARNING", "Evaluation at T_high=$T_high did not converge. Trying a slightly lower T_high.")
        ok_high, Δ_high = eval_gap_at(T_high - 1e-3)
        if !ok_high
            write_log(logFileName, "ERROR", "Couldn't evaluate Δ at T_high. Storing Tc = $T_high")
            open(fileName, "a") do io
                println(io, "$t2\t$T_high")
            end
            return
        else
            T_high -= 1e-3
        end
    end

    if Δ_high > gap_threshold
        write_log(logFileName, "WARNING", "Gap at initial T_high=$T_high is still > threshold. Increase T_high or adjust bracket.")
        # choose to set Tc = T_high here (or better: expand T_high and retry)
        open(fileName, "a") do io
            println(io, "$t2\t$T_high")
        end
        return
    end

    # Now true bisection: f(low)>0, f(high)<0
    iter = 0
    while (T_high - T_low) > T_tol && iter < max_iter
        T_try = 0.5 * (T_low + T_high)
        ok, Δ_try = eval_gap_at(T_try)
        if !ok
            # if solver fails at midpoint, move high bracket inward conservatively
            write_log(logFileName, "WARNING", "iter=$iter T_try=$T_try did not converge; shrinking high bound")
            T_high = T_try
            iter += 1
            continue
        end

        # --- append CSV log for this iteration (save progress) ---
        open(csvfile, "a") do io
            ts = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
            diff = T_high - T_low
            # iter, timestamp, T_high, T_low, T_try, converged(yes/no), Δ̄, T_diff
            println(io,
                "$iter,$ts," *
                @sprintf("%.6f", T_high) * "," *
                @sprintf("%.6f", T_low) * "," *
                @sprintf("%.6f", T_try) * "," *
                (ok ? "yes" : "no") * "," *
                @sprintf("%.4f", Δ_try) * "," *
                @sprintf("%.6f", diff)
            )
        end
        # ---------------------------------------------------------

        if Δ_try > gap_threshold
            # still superconducting at midpoint -> root (transition) is at higher T -> move low up
            T_low = T_try
        else
            # normal at midpoint -> transition lies at lower T -> move high down
            T_high = T_try
        end
        iter += 1
    end


    Tc = round(0.5 * (T_low + T_high), digits=3)
    open(fileName, "a") do io
        println(io, "$t2\t$Tc")
    end

    et = time() - st
    message = string("Completed in: ", format_elapsed_time(et))
    write_log(logFileName, "INFO", message)
    GC.gc()
end

timestart = time()
@everywhere begin
    df = CSV.read("$(dataSetFolder)df_square-octagon$N.csv", DataFrame)
    x = df.x
    y = df.y
    neighbors = hcat(df.n1, df.n2, df.n3, df.n4)
    # iso_map = compute_iso_map(df)
    iso_map = nothing
end

infoLogFileName = "$(logFolder)info.log"
pf = @__FILE__

# Detailed description of the program
desc = """
Calculates the critical temperature Tc for various values of hopping parameter t₂.
The program takes t2Vals as input and processes each value in parallel.
For each μ, the program calculates and stores the following data:
    - t2 : NNN hopping parameter
    - Tc : critical temperature
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

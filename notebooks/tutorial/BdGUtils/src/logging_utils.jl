module LoggingUtils
    using Dates
    using Printf

    export write_log
    export format_elapsed_time

    include("../src/general_utils.jl")
    using .GeneralUtils

    # retained for backward compatibility
    function write_log(filename::String, message::String, vars...)
        write_log(filename, "INFO", message, vars...)
    end

    # Enhanced function with log level and time format
    function write_log(filename::String, level::String, message::String, vars...; time_format="yyyy-mm-dd HH:MM:SS")
        try
            open(filename, "a") do file
                timestamp = Dates.format(Dates.now(), time_format)
                formatted_message = string("[$level] ", message, " ", join(vars, " "))
                write(file, "$timestamp: $formatted_message\n")
            end
        catch e
            println("Error writing to log file $filename: $e")
        end
    end

    """
        format_elapsed_time(elapsed_seconds)

    Format an elapsed time in seconds into seconds, minutes, or hours.

    Arguments:
    - `elapsed_seconds`: elapsed time in seconds

    Returns:
    - human-readable time string
    """
    function format_elapsed_time(elapsed_seconds)
        if elapsed_seconds < 60
            return string(round(elapsed_seconds, digits=2), " seconds")
        elseif elapsed_seconds < 3600
            return string(round(elapsed_seconds / 60, digits=2), " minutes")
        else
            return string(round(elapsed_seconds / 3600, digits=2), " hours")
        end
    end

end # Module end


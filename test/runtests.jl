using Test
using QuadGK

include("../src/general_utils.jl")
include("../src/logging_utils.jl")

using .GeneralUtils
using .LoggingUtils

include("test_tol.jl")
include("test_math_utils.jl")
include("test_misc.jl")

using Test

@testset "format_elapsed_time" begin

    @test format_elapsed_time(10.0) == "10.0 seconds"
    @test format_elapsed_time(59.999) == "60.0 seconds"

    @test format_elapsed_time(60.0) == "1.0 minutes"
    @test format_elapsed_time(120.0) == "2.0 minutes"

    @test format_elapsed_time(3600.0) == "1.0 hours"
    @test format_elapsed_time(7200.0) == "2.0 hours"

end

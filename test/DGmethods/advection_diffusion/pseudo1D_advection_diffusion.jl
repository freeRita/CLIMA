using MPI
using CLIMA
using Logging
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.DGmethods
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.LowStorageRungeKuttaMethod
using LinearAlgebra
using Printf
using Dates
using CLIMA.GenericCallbacks: EveryXWallTimeSeconds, EveryXSimulationSteps
using CLIMA.ODESolvers: solve!, gettime
using CLIMA.VTK: writevtk, writepvtu
using CLIMA.Mesh.Grids: EveryDirection, HorizontalDirection, VerticalDirection

if !@isdefined integration_testing
  const integration_testing =
    parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
end

const output = parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_OUTPUT","false")))

include("advection_diffusion_model.jl")

struct Pseudo1D{n, α, β, μ, δ} <: AdvectionDiffusionProblem end

function init_velocity_diffusion!(::Pseudo1D{n, α, β}, aux::Vars,
                                  geom::LocalGeometry) where {n, α, β}
  # Direction of flow is n with magnitude α
  aux.u = α * n

  # diffusion of strength β in the n direction
  aux.D = β * n * n'
end

function initial_condition!(::Pseudo1D{n, α, β, μ, δ}, state, aux, x,
                            t) where {n, α, β, μ, δ}
  ξn = dot(n, x)
  # ξT = SVector(x) - ξn * n
  state.ρ = exp(-(ξn - μ - α * t)^2 / (4 * β * (δ + t))) / sqrt(1 + t / δ)
end
function ∇initial_condition!(::Pseudo1D{n, α, β, μ, δ}, ∇state, aux, x,
                             t) where {n, α, β, μ, δ}
  ξn = dot(n, x)
  ∇state.ρ = -(2n * (ξn - μ - α * t) / (4 * β * (δ + t)) *
               exp(-(ξn - μ - α * t)^2 / (4 * β * (δ + t))) / sqrt(1 + t / δ))
end

function do_output(mpicomm, vtkdir, vtkstep, dg, Q, Qe, model, testname)
  ## name of the file that this MPI rank will write
  filename = @sprintf("%s/%s_mpirank%04d_step%04d",
                      vtkdir, testname, MPI.Comm_rank(mpicomm), vtkstep)

  statenames = flattenednames(vars_state(model, eltype(Q)))
  exactnames = statenames .* "_exact"

  writevtk(filename, Q, dg, statenames, Qe, exactnames)

  ## generate the pvtu file for these vtk files
  if MPI.Comm_rank(mpicomm) == 0
    ## name of the pvtu file
    pvtuprefix = @sprintf("%s/%s_step%04d", vtkdir, testname, vtkstep)

    ## name of each of the ranks vtk files
    prefixes = ntuple(MPI.Comm_size(mpicomm)) do i
      @sprintf("%s_mpirank%04d_step%04d", testname, i - 1, vtkstep)
    end

    writepvtu(pvtuprefix, prefixes, (statenames..., exactnames...))

    @info "Done writing VTK: $pvtuprefix"
  end
end


function run(mpicomm, ArrayType, dim, topl, N, timeend, FT, direction, dt,
             n, α, β, μ, δ, vtkdir, outputtime, fluxBC)

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = FT,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                         )
  model = AdvectionDiffusion{dim, fluxBC}(Pseudo1D{n, α, β, μ, δ}())
  dg = DGModel(model,
               grid,
               Rusanov(),
               CentralNumericalFluxDiffusive(),
               CentralNumericalFluxGradient(),
               direction=direction())

  Q = init_ode_state(dg, FT(0))

  lsrk = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0)

  eng0 = norm(Q)
  @info @sprintf """Starting
  norm(Q₀) = %.16e""" eng0

  # Set up the information callback
  starttime = Ref(now())
  cbinfo = EveryXWallTimeSeconds(60, mpicomm) do (s=false)
    if s
      starttime[] = now()
    else
      energy = norm(Q)
      @info @sprintf("""Update
                     simtime = %.16e
                     runtime = %s
                     norm(Q) = %.16e""", gettime(lsrk),
                     Dates.format(convert(Dates.DateTime,
                                          Dates.now()-starttime[]),
                                  Dates.dateformat"HH:MM:SS"),
                     energy)
    end
  end
  callbacks = (cbinfo,)
  if ~isnothing(vtkdir)
    # create vtk dir
    mkpath(vtkdir)

    vtkstep = 0
    # output initial step
    do_output(mpicomm, vtkdir, vtkstep, dg, Q, Q, model, "advection_diffusion")

    # setup the output callback
    cbvtk = EveryXSimulationSteps(floor(outputtime/dt)) do
      vtkstep += 1
      Qe = init_ode_state(dg, gettime(lsrk))
      do_output(mpicomm, vtkdir, vtkstep, dg, Q, Qe, model,
                "advection_diffusion")
    end
    callbacks = (callbacks..., cbvtk)
  end

  solve!(Q, lsrk; timeend=timeend, callbacks=callbacks)

  # Print some end of the simulation information
  engf = norm(Q)
  Qe = init_ode_state(dg, FT(timeend))

  engfe = norm(Qe)
  errf = euclidean_distance(Q, Qe)
  @info @sprintf """Finished
  norm(Q)                 = %.16e
  norm(Q) / norm(Q₀)      = %.16e
  norm(Q) - norm(Q₀)      = %.16e
  norm(Q - Qe)            = %.16e
  norm(Q - Qe) / norm(Qe) = %.16e
  """ engf engf/eng0 engf-eng0 errf errf / engfe
  errf
end

using Test
let
  CLIMA.init()
  ArrayType = CLIMA.array_type()

  mpicomm = MPI.COMM_WORLD
  ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
  loglevel = ll == "DEBUG" ? Logging.Debug :
  ll == "WARN"  ? Logging.Warn  :
  ll == "ERROR" ? Logging.Error : Logging.Info
  logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
  global_logger(ConsoleLogger(logger_stream, loglevel))

  polynomialorder = 4
  base_num_elem = 4

  expected_result = Dict()
  expected_result[2, 1, Float64, EveryDirection]      = 1.3278358805552103e-02
  expected_result[2, 2, Float64, EveryDirection]      = 1.5275567222209200e-03
  expected_result[2, 3, Float64, EveryDirection]      = 2.1158688830873719e-04
  expected_result[2, 4, Float64, EveryDirection]      = 5.4682730423646384e-06

  expected_result[2, 1, Float64, HorizontalDirection] = 4.7452978385924841e-02
  expected_result[2, 2, Float64, HorizontalDirection] = 4.2395734359794088e-03
  expected_result[2, 3, Float64, HorizontalDirection] = 7.5299237338321249e-05
  expected_result[2, 4, Float64, HorizontalDirection] = 4.5091115956660895e-07

  expected_result[2, 1, Float64, VerticalDirection]   = 4.7452978385924841e-02
  expected_result[2, 2, Float64, VerticalDirection]   = 4.2395734359794192e-03
  expected_result[2, 3, Float64, VerticalDirection]   = 7.5299237338312901e-05
  expected_result[2, 4, Float64, VerticalDirection]   = 4.5091115956933386e-07

  expected_result[3, 1, Float64, EveryDirection]      = 9.9996908954108274e-03
  expected_result[3, 2, Float64, EveryDirection]      = 7.8174266228124144e-04
  expected_result[3, 3, Float64, EveryDirection]      = 6.4581399695447892e-05
  expected_result[3, 4, Float64, EveryDirection]      = 3.8210741637228301e-06

  expected_result[3, 1, Float64, HorizontalDirection] = 1.8778435108867900e-02
  expected_result[3, 2, Float64, HorizontalDirection] = 2.1602914338589609e-03
  expected_result[3, 3, Float64, HorizontalDirection] = 2.9922904706653158e-04
  expected_result[3, 4, Float64, HorizontalDirection] = 7.7333058993042428e-06

  expected_result[3, 1, Float64, VerticalDirection]   = 6.7108645608372028e-02
  expected_result[3, 2, Float64, VerticalDirection]   = 5.9956622518388255e-03
  expected_result[3, 3, Float64, VerticalDirection]   = 1.0648920268019518e-04
  expected_result[3, 4, Float64, VerticalDirection]   = 6.3768467727359403e-07

  expected_result[2, 1, Float32, EveryDirection]      = 1.3278233818709850e-02
  expected_result[2, 2, Float32, EveryDirection]      = 1.5276039484888315e-03
  expected_result[2, 3, Float32, EveryDirection]      = 2.1158660820219666e-04

  expected_result[2, 1, Float32, HorizontalDirection] = 4.7453053295612335e-02
  expected_result[2, 2, Float32, HorizontalDirection] = 4.2393575422465801e-03
  expected_result[2, 3, Float32, HorizontalDirection] = 7.5274780101608485e-05

  expected_result[2, 1, Float32, VerticalDirection]   = 4.7453027218580246e-02
  expected_result[2, 2, Float32, VerticalDirection]   = 4.2393403127789497e-03
  expected_result[2, 3, Float32, VerticalDirection]   = 7.5348107202444226e-05

  expected_result[3, 1, Float32, EveryDirection]      = 9.9996607750654221e-03
  expected_result[3, 2, Float32, EveryDirection]      = 7.8179326374083757e-04
  expected_result[3, 3, Float32, EveryDirection]      = 6.4723266405053437e-05

  expected_result[3, 1, Float32, HorizontalDirection] = 1.8778320401906967e-02
  expected_result[3, 2, Float32, HorizontalDirection] = 2.1604031790047884e-03
  expected_result[3, 3, Float32, HorizontalDirection] = 2.9950391035526991e-04

  expected_result[3, 1, Float32, VerticalDirection]   = 6.7108325660228729e-02
  expected_result[3, 2, Float32, VerticalDirection]   = 5.9953825548291206e-03
  expected_result[3, 3, Float32, VerticalDirection]   = 1.1483333946671337e-04

  @testset "$(@__FILE__)" begin
    for FT in (Float64, Float32)
      numlevels = integration_testing || CLIMA.Settings.integration_testing ? (FT == Float64 ? 4 : 3) : 1
      result = zeros(FT, numlevels)
      for dim = 2:3
        for direction in (EveryDirection, HorizontalDirection,
                          VerticalDirection)
          for fluxBC in (true, false)
            if direction <: EveryDirection
              n = dim == 2 ? SVector{3, FT}(1/sqrt(2), 1/sqrt(2), 0) :
              SVector{3, FT}(1/sqrt(3), 1/sqrt(3), 1/sqrt(3))
            elseif direction <: HorizontalDirection
              n = dim == 2 ? SVector{3, FT}(1, 0, 0) :
              SVector{3, FT}(1/sqrt(2), 1/sqrt(2), 0)
            elseif direction <: VerticalDirection
              n = dim == 2 ? SVector{3, FT}(0, 1, 0) : SVector{3, FT}(0, 0, 1)
            end
            α = FT(1)
            β = FT(1 // 100)
            μ = FT(-1 // 2)
            δ = FT(1 // 10)
            for l = 1:numlevels
              Ne = 2^(l-1) * base_num_elem
              brickrange = ntuple(j->range(FT(-1); length=Ne+1, stop=1), dim)
              periodicity = ntuple(j->false, dim)
              bc = ntuple(j->(1,2), dim)
              topl = StackedBrickTopology(mpicomm, brickrange;
                                          periodicity = periodicity,
                                          boundary = bc)
              dt = (α/4) / (Ne * polynomialorder^2)
              @info "time step" dt

              timeend = 1
              outputtime = 1

              dt = outputtime / ceil(Int64, outputtime / dt)

              @info (ArrayType, FT, dim, direction, fluxBC)
              vtkdir = output ? "vtk_advection" *
                                "_poly$(polynomialorder)" *
                                "_dim$(dim)_$(ArrayType)_$(FT)_$(direction)" *
                                "_level$(l)" : nothing
              result[l] = run(mpicomm, ArrayType, dim, topl, polynomialorder,
                              timeend, FT, direction, dt, n, α, β, μ, δ, vtkdir,
                              outputtime, fluxBC)
              @test result[l] ≈ FT(expected_result[dim, l, FT, direction])
            end
            @info begin
              msg = ""
              for l = 1:numlevels-1
                rate = log2(result[l]) - log2(result[l+1])
                msg *= @sprintf("\n  rate for level %d = %e\n", l, rate)
              end
              msg
            end
          end
        end
      end
    end
  end
end

nothing

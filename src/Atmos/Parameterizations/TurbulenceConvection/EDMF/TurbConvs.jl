#### TurbConvs

export TurbConv
export get_ϕ_ψ

struct TurbConv{G, SVQ, SVT, TD, DT}
  grid::G
  q::SVQ
  q_new::SVQ
  tendencies::SVQ
  tmp::SVT
  tri_diag::TD
  tmp_O2::Dict
  dir_tree::DT
end

mutable struct TimeMarchingParams{FT}
  Δt::FT
  Δt_min::FT
  t_end::FT
  CFL::FT
  TimeMarchingParams(;Δt::FT,Δt_min::FT,t_end::FT,CFL::FT) where FT =
    new{FT}(Δt,Δt_min,t_end,CFL)
end

"""
    GridParams{FT,I}

Grid parameters used to initialize a grid.
"""
struct GridParams{FT,I}
  z_min::FT
  z_max::FT
  n_elems::I
  GridParams(;z_min::FT, z_max::FT, n_elems::I) where {FT,I} =
    new{FT,I}(z_min, z_max, n_elems)
end
import ..FiniteDifferenceGrids:Grid
Grid(gp::GridParams) = Grid(gp.z_min, gp.z_max, gp.n_elems)

"""
    get_ϕ_ψ(ϕ)

Convenience function to get individual
variables names from co-variance.
"""
function get_ϕ_ψ(ϕ)
  if ϕ == :cv_q_tot
    _ϕ = :q_tot; _ψ = :q_tot
  elseif ϕ == :cv_θ_liq
    _ϕ = :θ_liq; _ψ = :θ_liq
  elseif ϕ == :cv_θ_liq_q_tot
    _ϕ = :q_tot; _ψ = :θ_liq
  end
  return _ϕ, _ψ
end

function TurbConv(params, case::Case)
  @unpack params N_subdomains
  n_ud = N_subdomains-2

  grid = Grid(params[:GridParams])
  dd = DomainDecomp(gm=1,en=1,ud=n_ud)

  unkowns = (
  (:a     , DomainSubSet(gm=true,en=true,ud=true)),
  (:w     , DomainSubSet(gm=true,en=true,ud=true)),
  (:q_tot , DomainSubSet(gm=true,en=true,ud=true)),
  (:θ_liq , DomainSubSet(gm=true,en=true,ud=true)),
  (:tke   , DomainSubSet(en=true)),
  (:u     , DomainSubSet(gm=true)),
  (:v     , DomainSubSet(gm=true)),
  )

  tmp_vars = (
  (:ρ_0                    , DomainSubSet(gm=true)),
  (:p_0                    , DomainSubSet(gm=true)),
  (:T                      , DomainSubSet(gm=true,en=true,ud=true)),
  (:q_liq                  , DomainSubSet(gm=true,en=true,ud=true)),
  (:HVSD_a                 , DomainSubSet(ud=true)),
  (:HVSD_w                 , DomainSubSet(ud=true)),
  (:nh_press               , DomainSubSet(ud=true)),
  (:buoy                   , DomainSubSet(gm=true,en=true,ud=true)),
  (:∇buoyancy             , DomainSubSet(gm=true)),
  (:δ_model                , DomainSubSet(ud=true)),
  (:ε_model                , DomainSubSet(ud=true)),
  (:l_mix                  , DomainSubSet(gm=true)),
  (:K_m                    , DomainSubSet(gm=true)),
  (:K_h                    , DomainSubSet(gm=true)),
  (:α_0                    , DomainSubSet(gm=true)),
  (:q_tot_dry              , DomainSubSet(gm=true)),
  (:θ_dry                  , DomainSubSet(gm=true)),
  (:t_cloudy               , DomainSubSet(gm=true)),
  (:q_vap_cloudy           , DomainSubSet(gm=true)),
  (:q_tot_cloudy           , DomainSubSet(gm=true)),
  (:θ_cloudy               , DomainSubSet(gm=true)),
  (:CF                     , DomainSubSet(gm=true)),
  (:dTdt                   , DomainSubSet(gm=true)),
  (:dqtdt                  , DomainSubSet(gm=true)),
  (:mf_θ_liq               , DomainSubSet(gm=true)),
  (:mf_q_tot               , DomainSubSet(gm=true)),
  (:mf_tend_θ_liq          , DomainSubSet(gm=true)),
  (:mf_tend_q_tot          , DomainSubSet(gm=true)),
  (:mf_tmp                 , DomainSubSet(ud=true)),
  (:θ_ρ                    , DomainSubSet(gm=true)),
  (:ug                     , DomainSubSet(gm=true)),
  (:vg                     , DomainSubSet(gm=true)),
  (:subsidence             , DomainSubSet(gm=true)),
  )

  tri_diag_vars = (
  (:a     , DomainSubSet(gm=true)),
  (:b     , DomainSubSet(gm=true)),
  (:c     , DomainSubSet(gm=true)),
  (:f     , DomainSubSet(gm=true)),
  (:β     , DomainSubSet(gm=true)),
  (:γ     , DomainSubSet(gm=true)),
  (:xtemp , DomainSubSet(gm=true)),
  (:ρaK   , DomainSubSet(gm=true)),
  )

  q_2MO_vars = (
  (:values     , DomainSubSet(gm=true)),
  (:dissipation, DomainSubSet(gm=true)),
  (:entr_gain  , DomainSubSet(gm=true)),
  (:detr_loss  , DomainSubSet(gm=true)),
  (:buoy       , DomainSubSet(gm=true)),
  (:press      , DomainSubSet(gm=true)),
  (:shear      , DomainSubSet(gm=true)),
  (:interdomain, DomainSubSet(gm=true)),
  (:rain_src   , DomainSubSet(gm=true)),
  )

  q        = StateVec(unkowns, grid, dd)
  tmp      = StateVec(tmp_vars, grid, dd)
  tri_diag = StateVec(tri_diag_vars, grid, dd)
  q_new = deepcopy(q)
  tendencies = deepcopy(q)
  tmp_O2   = Dict()
  tmp_O2[:tke] = StateVec(q_2MO_vars, grid, dd)
  dir_tree = DirTree(string(case), Tuple([name for (name, nsd) in unkowns]))

  turb_conv = TurbConv(grid, q, q_new, tendencies, tmp, tri_diag, tmp_O2, dir_tree)
end

#### Surface functions

using CLIMA.SurfaceFluxes.Nishizawa2018
using Distributions
using Statistics

export update_surface!

"""
    update_surface!

Update surface conditions including
 - `windspeed`
 - `ρq_tot_flux`
 - `ρθ_liq_flux`
 - `bflux`
 - `obukhov_length`
 - `rho_uflux`
 - `rho_vflux`
"""
function update_surface! end

abstract type SurfaceType end
struct SurfaceFixedFlux{FT} <: SurfaceType
  T::FT
  P::FT
  q_tot::FT
  shf::FT
  lhf::FT
  Tsurface::FT
  ρ_0_surf::FT
  α_0_surf::FT
  ρq_tot_flux::FT
  ρθ_liq_flux::FT
  ustar::FT
  windspeed_min::FT
  tke_tol::FT
  area::FT
  function SurfaceFixedFlux(;T, P, q_tot, ustar, windspeed_min, tke_tol, area)
    FT = typeof(T)
    q_pt = PhasePartition(q_tot)
    ρ_0_surf = air_density(T, P, q_pt)
    α_0_surf = 1/ρ_0_surf
    Tsurface = 299.1 * exner(P)
    lhf = 5.2e-5 * ρ_0_surf * latent_heat_vapor(Tsurface)
    shf = 8.0e-3 * cp_m(q_pt) * ρ_0_surf
    ρ_tflux =  shf / cp_m(q_pt)
    ρq_tot_flux = lhf / latent_heat_vapor(Tsurface)
    ρθ_liq_flux = ρ_tflux / exner(P)
    return new{FT}(T, P, q_tot, shf, lhf, Tsurface,
      ρ_0_surf, α_0_surf, ρq_tot_flux, ρθ_liq_flux, ustar, windspeed_min, tke_tol, area)
  end
end

function update_surface!(tmp::StateVec, q::StateVec, grid::Grid{FT}, params, model::SurfaceFixedFlux) where FT
  gm, en, ud, sd, al = allcombinations(tmp)
  k_1 = first_interior(grid, Zmin())
  z_1 = grid.zc[k_1]
  T_1 = tmp[:T, k_1, gm]
  θ_liq_1 = q[:θ_liq, k_1, gm]
  q_tot_1 = q[:q_tot, k_1, gm]
  v_1 = q[:v, k_1, gm]
  u_1 = q[:u, k_1, gm]

  params[:windspeed] = compute_windspeed(q, k_1, gm, FT(0.0))
  params[:bflux] = buoyancy_flux(model.shf, model.lhf, T_1, q_tot_1, model.α_0_surf)

  params[:obukhov_length] = compute_MO_len(model.ustar, params[:bflux])
  params[:rho_uflux] = - model.ρ_0_surf *  model.ustar * model.ustar / params[:windspeed] * u_1
  params[:rho_vflux] = - model.ρ_0_surf *  model.ustar * model.ustar / params[:windspeed] * v_1
end


function percentile_bounds_mean_norm(low_percentile::FT, high_percentile::FT, n_samples::I) where {FT<:Real, I}
    x = rand(Normal(), n_samples)
    xp_low = quantile(Normal(), low_percentile)
    xp_high = quantile(Normal(), high_percentile)
    filter!(y -> xp_low<y<xp_high, x)
    return Statistics.mean(x)
end

"""
    surface_tke(ustar::FT, wstar::FT, zLL::FT, obukhov_length::FT) where FT<:Real

computes the surface tke

 - `ustar` friction velocity
 - `wstar` convective velocity
 - `zLL` elevation at the first grid level
 - `obukhov_length` Monin-Obhukov length
"""
function surface_tke(ustar::FT, wstar::FT, zLL::FT, obukhov_length::FT) where FT<:Real
  if unstable(obukhov_length)
    return ((3.75 + cbrt(zLL/obukhov_length * zLL/obukhov_length)) * ustar * ustar + 0.2 * wstar * wstar)
  else
    return (3.75 * ustar * ustar)
  end
end

"""
    surface_variance(flux1::FT, flux2::FT, ustar::FT, zLL::FT, obukhov_length::FT) where FT<:Real

computes the surface variance given

 - `ustar` friction velocity
 - `wstar` convective velocity
 - `zLL` elevation at the first grid level
 - `obukhov_length` Monin-Obhukov length
"""
function surface_variance(flux1::FT, flux2::FT, ustar::FT, zLL::FT, obukhov_length::FT) where FT<:Real
  c_star1 = -flux1/ustar
  c_star2 = -flux2/ustar
  if unstable(obukhov_length)
    return 4 * c_star1 * c_star2 * (1 - 8.3 * zLL/obukhov_length)^(-2/3)
  else
    return 4 * c_star1 * c_star2
  end
end

"""
    compute_convective_velocity(bflux, inversion_height)

Computes the convective velocity scale, given the buoyancy flux
`bflux`, and inversion height `inversion_height`.
FIXME: add reference
"""
compute_convective_velocity(bflux::FT, inversion_height::FT) where FT = cbrt(max(bflux * inversion_height, FT(0)))

"""
    compute_windspeed(q::StateVec, k::I, i::I, windspeed_min::FT)

Computes the windspeed
"""
function compute_windspeed(q::StateVec, k::I, i::I, windspeed_min::FT) where {FT, I}
  return max(hypot(q[:u, k, i], q[:v, k, i]), windspeed_min)
end

"""
    compute_inversion_height(tmp::StateVec, q::StateVec, grid::Grid, params)

Computes the inversion height (a non-local variable)
FIXME: add reference
"""
function compute_inversion_height(tmp::StateVec, q::StateVec, grid::Grid, params)
  @unpack params Ri_bulk_crit SurfaceType
  gm, en, ud, sd, al = allcombinations(q)
  k_1 = first_interior(grid, Zmin())
  windspeed = compute_windspeed(q, k_1, gm, 0.0)^2

  # test if we need to look at the free convective limit
  z = grid.zc
  h = 0
  Ri_bulk, Ri_bulk_low = 0, 0
  ts = ActiveThermoState(q, tmp, k_1, gm)
  θ_ρ_b = virtual_pottemp(ts)
  k_star = k_1
  if windspeed <= SurfaceType.tke_tol
    for k in over_elems_real(grid)
      if tmp[:θ_ρ, k] > θ_ρ_b
        k_star = k
        break
      end
    end
    h = (z[k_star] - z[k_star-1])/(tmp[:θ_ρ, k_star] - tmp[:θ_ρ, k_star-1]) * (θ_ρ_b - tmp[:θ_ρ, k_star-1]) + z[k_star-1]
  else
    for k in over_elems_real(grid)
      Ri_bulk_low = Ri_bulk
      Ri_bulk = grav * (tmp[:θ_ρ, k] - θ_ρ_b) * z[k]/θ_ρ_b / (q[:u, k, gm]^2 + q[:v, k, gm]^2)
      if Ri_bulk > Ri_bulk_crit
        k_star = k
        break
      end
    end
    h = (z[k_star] - z[k_star-1])/(Ri_bulk - Ri_bulk_low) * (Ri_bulk_crit - Ri_bulk_low) + z[k_star-1]
  end
  return h
end

"""
    compute_MO_len(ustar::FT, bflux::FT) where {FT<:Real}

Compute Monin-Obhukov length given
 - `ustar` friction velocity
 - `bflux` buoyancy flux
"""
function compute_MO_len(ustar::FT, bflux::FT) where {FT<:Real}
  return abs(bflux) < FT(1e-10) ? FT(0) : -ustar * ustar * ustar / bflux / k_Karman
end

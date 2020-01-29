export @uaware, urule, AbstractUnitCtx, DriverUnitCtx

using MacroTools
using MacroTools: postwalk, @capture, @expand, prettify

abstract type AbstractUnitCtx end
struct DriverUnitCtx <: AbstractUnitCtx end

# Drivers will specialise urule(::DriverUnitCtx)
urule(::AbstractUnitCtx) = false
# Enable unit types in structures
urule(::DriverUnitCtx) = true

function substitution(Ctx, ex)
  if @capture(ex, f_Symbol_U(p1_,p2_))
    return urule(Ctx) ? :(units($p1,$p2)) : p1
  end
  ex
end

macro uaware(Ctx::AbstractUnitCtx, ex)
  Ctx = DriverUnitCtx()
  @capture(ex, struct _ __ end) || error("uaware only supports structures.")
  p = postwalk(x -> substitution(Ctx, x) , ex)
  @show prettify(p), urule(Ctx)
  esc(p)
end

"""
  Apply global unit annotation rule, supplied by evaluating the method
  `urule(::DriverUnitCtx)`.

  This macro is most useful when defining structures or methods that are unit aware, however
  may still be invoked on drivers which do not support units.
"""
macro uaware(ex)
  Ctx = DriverUnitCtx()
  esc(macroexpand(__module__, :(UnitAnnotations.@uaware($Ctx, $ex))))
end

# Quick test
# @uaware struct Foo{FT}
#   x::U(FT,:massflux)
# end
#
# Foo{Float64}(1.0e0u"kg/m^2/s")

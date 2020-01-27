using MacroTools

abstract type AbstractUnitCtx end
struct DriverUnitCtx end

# Drivers will specialise urule(::DriverUnitCtx)
@inline function urule(::AbstractUnitCtx) = false

"""
  Apply global unit annotation rule, supplied by evaluating the method
  `urule(::DriverUnitCtx)`.

  This macro is most useful when defining structures or methods that are unit aware, however
  may still be invoked on drivers which do not support units.
"""
macro uaware(ex)
  uaware(DriverUnitCtx(), ex)
end

macro uaware(Ctx::AbstractUnitCtx, ex)
  # TODO: define a traversal rule. Will return the resultant expression
  postwalk()
end

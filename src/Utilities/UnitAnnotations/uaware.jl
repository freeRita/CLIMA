using MacroTools
using MacroTools: postwalk, @capture, @expand, prettify

abstract type AbstractUnitCtx end
struct DriverUnitCtx <: AbstractUnitCtx end

# Drivers will specialise urule(::DriverUnitCtx)
@inline urule(::AbstractUnitCtx) = false

macro uaware(Ctx::AbstractUnitCtx, ex)
  @capture(ex, struct _ __ end) || error("uaware only supports structures.")
  p = postwalk(x -> substitution(Ctx, x) , ex)
  @show ex, prettify(p)
  p
end

"""
  Apply global unit annotation rule, supplied by evaluating the method
  `urule(::DriverUnitCtx)`.

  This macro is most useful when defining structures or methods that are unit aware, however
  may still be invoked on drivers which do not support units.
"""
macro uaware(ex)
  Ctx = DriverUnitCtx()
  macroexpand(__module__, :(UnitAnnotations.@uaware($Ctx, $ex)))
end

function substitution(Ctx, ex)
  if @capture(ex, f_Symbol_U(p1_,p2_))
    return :(begin
      union = U($p1, $p2)
      if union.a <: Quantity
        return $(urule(Ctx) ? :(union.a) : :(union.b))
      end
      $(urule(Ctx) ? :(union.b) : :(union.a))
    end)
  end
  ex
end

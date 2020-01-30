using MacroTools
using MacroTools: postwalk, @capture, @expand, prettify

function substitution(Ctx, ex)
  if @capture(ex, f_Symbol_U(p1_,p2_))
    return :(urule($Ctx) ? units($p1,$p2) : $p1)
  end
  ex
end

"""
  Apply global unit annotation rule, supplied by evaluating the method
  `urule(::DriverUnitCtx)`.

  This macro is most useful when defining structures or methods that are unit aware, however
  may still be invoked on drivers which do not support units.
"""
macro uaware(ex)
  @capture(ex, struct sig_ attrs__ end) || error("@uaware must be applied to a struct definiton")

  # Rename the struct
  local sig_unitful
  local name
  if @capture(sig, name_{p1__} <: sname_{p2__})
    sig_unitful = :($(Symbol(:U, name)){$(p1...)} <: $sname{$(p2...)})
  elseif @capture(sig, name_{p1__})
    sig_unitful = :($(Symbol(:U, name)){$(p1...)})
  elseif @capture(sig, name_)
    sig_unitful = Symbol(:U, name)
  end
  mname = Symbol(:U, name)

  # First just remove all units
  unitless = postwalk(ex) do x
    @capture(x, _Symbol_U(p1_, _)) && (return p1)
    x
  end

  take_unit(expr) = postwalk(expr) do x
    @capture(x, _Symbol_U(p1_, p2_)) && (return :(units($p1, $p2)))
    x
  end

  # Now make a version with unit annotations
  unitful = quote
    struct $sig_unitful
      $(map(take_unit, attrs)...)
    end
  end

  # Lastly provide a unifying compiler

  q = quote
    $unitless
    $unitful
  end
  @show prettify(q)
  esc(q)
end

# Quick test
@uaware struct Foo{FT}
  x::U(FT,:massflux)
end
#
# Foo{Float64}(1.0e0u"kg/m^2/s")

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

  # Remove docstrings from attrs
  attrs = filter(x->!(x isa String), attrs)

  # Rename the struct
  local sig_unitless
  local sig_unitful
  local name
  local union
  local p1
  if @capture(sig, name_{p1__} <: sname_{p2__})
    lname = Symbol(:L, name)
    uname = Symbol(:U, name)
    sig_unitless = :($lname{$(p1...)} <: $sname{$(p2...)})
    sig_unitful  = :($uname{$(p1...)} <: $sname{$(p2...)})
    union = :($name{$(p1...)} = Union{$lname{$(p1...)}, $uname{$(p1...)}})
  elseif @capture(sig, name_{p1__})
    lname = Symbol(:L, name)
    uname = Symbol(:U, name)
    sig_unitless = :($lname{$(p1...)})
    sig_unitful  = :($uname{$(p1...)})
    union = :($name{$(p1...)} = Union{$lname{$(p1...)}, $uname{$(p1...)}})
  elseif @capture(sig, name_)
    sig_unitless = Symbol(:L, name)
    sig_unitful  = Symbol(:U, name)
    union = :($name = Union{$sig_unitless, $sig_unitful})
  end
  @isdefined(p1) || (p1 = Any[])
  @assert sig_unitful !== sig_unitless

  # First just remove all units
  remove_unit(expr) = postwalk(expr) do x
    @capture(x, _Symbol_U(FT_, _)) && (return FT)
    x
  end

  attrs_unitless = map(remove_unit, attrs)
  unitless = quote
    struct $sig_unitless
      $(attrs_unitless...)
    end
  end

  # Now make a version with unit annotations
  take_unit(expr) = postwalk(expr) do x
    @capture(x, _Symbol_U(FT_, usym_)) && (return :(units($FT, $usym)))
    x
  end

  attrs_unitful = map(take_unit, attrs)
  unitful = quote
    struct $sig_unitful
      $(attrs_unitful...)
    end
  end

  # Lastly provide the constructors
  constr_unitless = quote
    function (::typeof($name{$(p1...)}))($(attrs_unitless...)) where {$(p1...)}
      $lname{$(p1...)}($(attrs_unitless...))
    end
  end
  constr_unitful = quote
    function (::typeof($name{$(p1...)}))($(attrs_unitful... )) where {$(p1...)}
      $uname{$(p1...)}($(attrs_unitful...))
    end
  end

  q = quote
    $unitless
    $unitful
    $union
    $constr_unitless
    $constr_unitful
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

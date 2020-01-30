using MacroTools
using MacroTools: postwalk, @capture, @expand, prettify

"""
  Duplicate structure definitions, maintaining a common constructor, to enable optional
  unit annotation.

  This macro is most useful when defining structures or methods that are unit aware, however
  may still be invoked on drivers which do not support units.
"""
macro uaware(ex)
  @capture(ex, struct sig_ attrs__ end) || error("@uaware must be applied to a struct definiton")

  # Remove docstrings from attrs
  attrs = filter(x->!(x isa String), attrs)

  # Rename the struct
  local sig_unitless, sig_unitful, name, union, p1, p2
  if @capture(sig, name_{p1__} <: sname_{p2__})
    lname = Symbol(:L, name)
    uname = Symbol(:U, name)
    sig_unitless = :($lname{$(p1...)} <: $sname{$(p2...)})
    sig_unitful  = :($uname{$(p1...)} <: $sname{$(p2...)})
    union = :($name{$(p1...)} = Union{$lname{$(p1...)}, $uname{$(p1...)}})
  elseif @capture(sig, name_{p1__} <: sname_)
    p2 = Any[]
    lname = Symbol(:L, name)
    uname = Symbol(:U, name)
    sig_unitless = :($lname{$(p1...)} <: $sname)
    sig_unitful  = :($uname{$(p1...)} <: $sname)
    union = :($name{$(p1...)} = Union{$lname{$(p1...)}, $uname{$(p1...)}})
  elseif @capture(sig, name_{p1__})
    lname = Symbol(:L, name)
    uname = Symbol(:U, name)
    sig_unitless = :($lname{$(p1...)})
    sig_unitful  = :($uname{$(p1...)})
    union = :($name{$(p1...)} = Union{$lname{$(p1...)}, $uname{$(p1...)}})
  elseif @capture(sig, name_)
    lname = Symbol(:L, name)
    uname = Symbol(:U, name)
    p1, p2 = Any[], Any[]
    sig_unitless = lname
    sig_unitful  = uname
    union = :($name = Union{$sig_unitless, $sig_unitful})
  end
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

  params_unitless = filter(x->!(x isa String), attrs_unitless)
  params_unitful  = filter(x->!(x isa String), attrs_unitful)

  # Lastly provide the constructors
  constr_unitless = quote
    function (::typeof($name{$(p1...)}))($(params_unitless...)) where {$(p1...)}
      $lname{$(p1...)}($(attrs_unitless...))
    end
  end
  constr_unitful = quote
    function (::typeof($name{$(p1...)}))($(params_unitful... )) where {$(p1...)}
      $uname{$(p1...)}($(attrs_unitful...))
    end
  end

  return quote
    Base.@__doc__ $unitless
    Base.@__doc__ $unitful
    Base.@__doc__ $union
    Base.@__doc__ $constr_unitless
    Base.@__doc__ $constr_unitful
  end |> esc
end

# Quick test
@uaware struct Foo{FT}
  x::U(FT,:massflux)
end
#
# Foo{Float64}(1.0e0u"kg/m^2/s")

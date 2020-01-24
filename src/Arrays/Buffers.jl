module Buffers

using Requires

export Buffer
export SingleBuffer, DoubleBuffer
export get_stage, get_transfer, prepare_transfer!, prepare_stage!

@enum BufferKind begin
  SingleBuffer
  DoubleBuffer
end

###
# Note: We use pinned and device-mapped hostbuffers in double staging
# Potential improvements
# - Investigate if we need to device-map
# - Implement single buffered with pinned + device-mapped hostbuffers

"""
    Buffer{T}(::Type{Arr}, kind, dims...; pinned = true)

Optionally double buffered space for MPI communication.
Call `get_stage()` to get a buffer into which to stage
data for communication.
Call `get_transfer()` to get the buffer
to use for communication.

In the case that double buffering is not needed, we avoid
the extra copy and use the staging buffer for transfers.

If we're running on the host only, double buffering is not
necessary. If we're running on the GPU and we have
CUDA-enabled MPI, we can avoid double buffering and use the
GPU memory directly. If we do not have CUDA-enabled MPI, we
must use double buffering, where the staging buffer is on
the device and the transfer buffer is on the host.

# Arguments
- `T`: Element type
- `Arr::Type`: What kind of array to allocate for `stage`
- `kind::BufferKind`: Either `Single` or `Double`
- `dims...`: Dimensions of the array

# Keyword Arguments
- `pinned`:  Register the `transfer` buffer with CUDA
"""
struct Buffer{T, Arr, Buff}
  stage :: Arr     # Same type as Q.data, used for staging
  transfer :: Buff # Union{Nothing,Buff}

  function Buffer{T}(::Type{Arr}, kind, dims...; pinned = true) where {T, Arr}
    if kind == SingleBuffer
      transfer = nothing
    elseif kind == DoubleBuffer
      transfer = zeros(T, dims...)
      if pinned
        # XXX: make this
        # transfer = register(transfer) to obtain a HostBuffer
        # or try:
        # ```
        #   transfer = register(transfer)::HostBuffer
        #   stage    = # CuArray(transfer)
        # ```
        register(transfer)
        # XXX: Should this finalizer be attached to the Buffer struct?
        finalizer(unregister, transfer)
      end
    else
      error("Bufferkind $kind is not implemented yet")
    end

    stage = similar(Arr, T, dims...)
    buffer = new{T, typeof(stage), typeof(transfer)}(stage, transfer)

    return buffer
  end
end

function get_stage(buf::Buffer)
  return buf.stage
end

function get_transfer(buf::Buffer)
  if buf.transfer === nothing
    return buf.stage
  else
    return buf.transfer
  end
end

function prepare_transfer!(buf::Buffer)
  if buf.transfer === nothing
    # nothing to do here
  else
    # XXX: async
    copy!(buf.transfer, buf.stage)
  end
end

function prepare_stage!(buf::Buffer)
  if buf.transfer === nothing
    # nothing to do here
  else
    # XXX: async
    copy!(buf.stage, buf.transfer)
  end
end

######
# Internal methods
######
register(x) = nothing
unregister(x) = nothing


@init @require CUDAdrv = "c5f51814-7f29-56b8-a69c-e4d8f6be1fde" begin
  using .CUDAdrv
  import .CUDAdrv.Mem

  function register(arr)
    if sizeof(arr) == 0
      @warn "Array is of size 0. Can't pin register array with CUDA" size(arr) typeof(arr)
      return arr
    end
    GC.@preserve arr begin
      Mem.register(Mem.HostBuffer, pointer(arr), sizeof(arr), Mem.HOSTREGISTER_DEVICEMAP)
    end
  end

  function unregister(arr)
    if sizeof(arr) == 0
      return
    end
    GC.@preserve arr begin
      ptr = pointer(arr)
      buf = Mem.HostBuffer(ptr, sizeof(arr), CuCurrentContext(), true)
      # XXX: CUDAdrv.Mem.unregister is only implemented for HostBuffer
      Mem.unregister(buf)
    end
  end
end

end #module
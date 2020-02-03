module Buffers

using Requires
using MPI
using CUDAapi

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

  function Buffer{T}(::Type{Arr}, kind, dims...; pinned = false) where {T, Arr}
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

###
# By design we intend to run async copies on the default stream
# to obtain concurrency computation should run on a non-blocking
# secondary stream. This is due to the fact that CUDA-aware MPI
# will run operations on the default stream and those operations
# are implicitly synchronising.
#
# In a double buffering scenario we want the memcpy to be asynchronous
# with respect to other streams, but we need to synchronize the default
# stream before the ISend
###

function prepare_transfer!(buf::Buffer)
  if buf.transfer === nothing
    # nothing to do here
  else
    copybuffer!(buf.transfer, buf.stage, async=false)
  end
end

function prepare_stage!(buf::Buffer)
  if buf.transfer === nothing
    # nothing to do here
  else
    copybuffer!(buf.stage, buf.transfer, async=true)
  end
end

######
# Internal methods
######
register(x) = nothing
unregister(x) = nothing

"""
  copybuffer(A, B; async=true)

Copy a buffer from device to host or vice-versa. Internally this uses
`cudaMemcpyAsync` on the `CuDefaultStream`. The keyword argument 
`async` determines whether it is asynchronous with regard to the host.
"""
function copybuffer!(A::AbstractArray, B::AbstractArray; async=true)
  copy!(A, B)
end


@init @require CUDAdrv = "c5f51814-7f29-56b8-a69c-e4d8f6be1fde" @require CuArrays = "3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
  using .CUDAdrv
  using .CuArrays

  import CUDAdrv.Mem

  function register(arr)
    if sizeof(arr) == 0
      @warn "Array is of size 0. Can't pin register array with CUDA" size(arr) typeof(arr)
      return arr
    end
    GC.@preserve arr begin
      # XXX: is Mem.HOSTREGISTER_DEVICEMAP necessary?
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


  # CUDAdrv.jl throws on CUDA_ERROR_NOT_READY
  function queryStream(hStream)
      err = CUDAapi.@runtime_ccall((:cuStreamQuery, CUDAdrv.libcuda), CUDAdrv.CUresult,
                          (CUDAdrv.CUstream,),
                          hStream)

      if err === CUDAdrv.CUDA_ERROR_NOT_READY
          return false
      elseif err === CUDAdrv.CUDA_SUCCESS
          return true
      else
          CUDAdrv.throw_api_error(err)
      end
  end

  """
      friendlysynchronize(stream)

  MPI defines a notion of progress which means that MPI operations
  need the program to call MPI functions (potentially multiple times)
  to make progress and eventually complete. In some implementations,
  progress on one rank may need MPI to be called on another rank.

  As a result blocking by for example calling cudaStreamSynchronize,
  may create a deadlock in some cases because not calling MPI will
  not make other ranks progress.
  """
  function friendlysynchronize(stream)
    status = false
    while !status
      status = queryStream(stream)
      MPI.Iprobe(MPI.MPI_ANY_SOURCE, MPI.MPI_ANY_TAG, MPI.COMM_WORLD)
    end
    return
  end

  function async_copy!(A, B, N, stream)
    GC.@preserve A B begin
      ptrA = pointer(A)
      ptrB = pointer(B)
      unsafe_copyto!(ptrA, ptrB, N, async=true, stream=stream)
    end
  end
  function copybuffer!(A::Array, B::CuArray; async=true)
    @assert sizeof(A) == sizeof(B)
    stream = CuDefaultStream()
    async_copy!(A, B.buf, sizeof(A), stream)
    if !async
      friendlysynchronize(stream)
    end
  end
  function copybuffer!(A::CuArray, B::Array; async=true)
    @assert sizeof(A) == sizeof(B)
    stream = CuDefaultStream()
    async_copy!(A.buf, B, sizeof(A), stream)
    if !async
      friendlysynchronize(stream)
    end
  end
end

end #module


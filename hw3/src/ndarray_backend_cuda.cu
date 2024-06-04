#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <float.h>

#include <iostream>
#include <sstream>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides

__device__ size_t getPosition(const size_t size, const CudaVec& shape, const CudaVec& strides, size_t offset, size_t gid){
  size_t dim = shape.size;
  size_t position = offset;
  size_t compact_stride = size;
  for(size_t i=0;i<dim;++i){
    compact_stride /= shape.data[i];
    size_t idx = gid / compact_stride;
    position += idx * strides.data[i];
    gid = gid % compact_stride;
  }
  return position;
}

__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN SOLUTION
  if(gid<size){
    out[gid] = a[getPosition(size, shape, strides, offset, gid)];
  }
  /// END SOLUTION
}

__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                                  CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if(gid < size){
    out[getPosition(size, shape, strides, offset, gid)] = a[gid];
  }
}

__global__ void ScalarSetitemKernel(const scalar_t val, scalar_t* out, size_t size, CudaVec shape,
  CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if(gid < size){
    out[getPosition(size, shape, strides, offset, gid)] = val;
  }
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}



void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
  /// END SOLUTION
}



void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(val, out->ptr, size, VecToCuda(shape),
                                        VecToCuda(strides), offset);
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */


////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////
#define EwiseOpKernel(name, op)\
__global__ void Ewise##name##Kernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size){\
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;\
  if(gid<size) {\
    out[gid] = a[gid] op b[gid]; \
  }\
}\

#define ScalarOpKernel(name, op)\
__global__ void Scalar##name##Kernel(const scalar_t* a, const scalar_t val, scalar_t* out, size_t size){\
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;\
  if(gid<size) {\
    out[gid] = a[gid] op val; \
  }\
}\

#define EwiseOp(name)\
void Ewise##name(const CudaArray& a, const CudaArray& b, CudaArray* out){ \
  CudaDims dim = CudaOneDim(out->size); \
  Ewise##name##Kernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, a.size);\
}                                                                            \

#define ScalarOp(name) \
void Scalar##name(const CudaArray& a, scalar_t val, CudaArray* out){   \
  CudaDims dim = CudaOneDim(out->size); \
  Scalar##name##Kernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, a.size);\
}                                                                            \


EwiseOpKernel(Mul, *)
ScalarOpKernel(Mul, *)
EwiseOpKernel(Div, /)
ScalarOpKernel(Div, /)

EwiseOp(Mul)
ScalarOp(Mul)
EwiseOp(Div)
ScalarOp(Div)


__global__ void ScalarPowerKernel(const scalar_t* a, const scalar_t val, scalar_t* out, size_t size){
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if(gid<size) {
    out[gid] = pow(a[gid] , val); 
  }
}

__global__ void EwiseMaximumKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size){
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if(gid<size) {
    out[gid] = max(a[gid] , b[gid]); 
  }
}

__global__ void ScalarMaximumKernel(const scalar_t* a, const scalar_t val, scalar_t* out, size_t size){
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if(gid<size) {
    out[gid] = max(a[gid] , val); 
  }
}
ScalarOp(Power)
ScalarOp(Maximum)
EwiseOp(Maximum)


#define EwiseCmpKernel(name, op)\
__global__ void Ewise##name##Kernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size){\
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;\
  if(gid<size) {\
    out[gid] = (a[gid] op b[gid])?1:0; \
  }\
}\

#define ScalarCmpKernel(name, op)\
__global__ void Scalar##name##Kernel(const scalar_t* a, const scalar_t val, scalar_t* out, size_t size){\
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;\
  if(gid<size) {\
    out[gid] = (a[gid] op val)?1:0; \
  }\
}\

#define EwiseCmp(name)\
void Ewise##name(const CudaArray& a, const CudaArray& b, CudaArray* out){ \
  CudaDims dim = CudaOneDim(out->size); \
  Ewise##name##Kernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr,a.size);\
}                                                                            \

#define ScalarCmp(name) \
void Scalar##name(const CudaArray& a, scalar_t val, CudaArray* out){   \
  CudaDims dim = CudaOneDim(out->size); \
  Scalar##name##Kernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, a.size);\
}                                                                            \


EwiseCmpKernel(Eq, ==)
EwiseCmpKernel(Ge, >=)
ScalarCmpKernel(Eq, ==)
ScalarCmpKernel(Ge, >=)
EwiseCmp(Eq)
EwiseCmp(Ge)
ScalarCmp(Eq)
ScalarCmp(Ge)

#define EwiseFuncKernel(name, func)\
__global__ void Ewise##name##Kernel(const scalar_t* a, scalar_t* out, size_t size){\
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;\
  if(gid<size) {\
    out[gid] = func(a[gid]); \
  }\
}\

#define EwiseFunc(name)                                \
void Ewise##name(const CudaArray& a, CudaArray* out){   \
  CudaDims dim = CudaOneDim(out->size); \
  Ewise##name##Kernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size);\
}                                                             \

EwiseFuncKernel(Log, log)
EwiseFuncKernel(Exp, exp)
EwiseFuncKernel(Tanh, tanh)
EwiseFunc(Log)
EwiseFunc(Exp)
EwiseFunc(Tanh)

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     uint32_t numARows, uint32_t numAColumns,
                                     uint32_t numBRows, uint32_t numBColumns,
                                     uint32_t numCRows, uint32_t numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  __shared__ float tailed_A[TILE][TILE];
  __shared__ float tailed_B[TILE][TILE];
  
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int idx = blockIdx.x * TILE + threadIdx.x;
  int idy = blockIdx.y * TILE + threadIdx.y;
  float PValues = 0;
  for(int i=0; i<numAColumns; i+= TILE){
    if(i + tx < numAColumns && idy < numCRows){
      tailed_A[ty][tx] = A[idy*numAColumns + i + tx]; // A[idy][i+tx]
    }
    if(i+ty < numAColumns && idx < numCColumns){
      tailed_B[ty][tx] = B[(i+ty)*numBColumns + idx]; // B[(i+ty)][idx]
    }
    __syncthreads();
    if(idx < numCColumns && idy < numCRows){
      for(int k=0; k<TILE && i+k < numAColumns; ++k){
        PValues += tailed_A[ty][k] * tailed_B[k][tx];
      }
    }
    __syncthreads();
    
  }
  if(idx < numCColumns && idy < numCRows){
    C[idy * numCColumns + idx] = PValues;
  }
}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   * 
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */
  /// BEGIN SOLUTION
  auto ceil = [](int a, int b){
    return int((a + b - 1) / b);
  };
  int BLOCK_SIZE_X = TILE;
  int BLOCK_SIZE_Y = TILE;
  //int BLOCK_SIZE = BLOCK_SIZE_X * BLOCK_SIZE_Y;
  dim3 DimGrid(ceil(out->size, BLOCK_SIZE_X), 
    ceil(out->size, BLOCK_SIZE_Y), 1);
  dim3 DimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
  matrixMultiplyShared<<<DimGrid, DimBlock>>>(a.ptr, b.ptr, out->ptr,M,N,N,P,M,P);
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////
__global__ void ReduceMaxKernel(const scalar_t* a, scalar_t * out, size_t reduce_size, size_t len){
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if(gid < len){
    size_t begin = reduce_size * gid;
    scalar_t res = a[begin];
    for(size_t i=1; i<reduce_size;++i){
      res = max(res, a[begin+i]);
    }
    out[gid] = res;
  }
}

__global__ void ReduceSumKernel(const scalar_t* a, scalar_t * out, size_t reduce_size, size_t len){
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if(gid < len){
    size_t begin = reduce_size * gid;
    scalar_t res = a[begin];
    for(size_t i=1;i<reduce_size;++i){
      res += a[begin+i];
    }
    out[gid] = res;
  }
}
void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out->size);
  /// END SOLUTION
}



void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
   * can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out->size);
  /// END SOLUTION
}

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}

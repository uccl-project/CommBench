/* Stub for CUDA compilation on aarch64: skips ARM GCC built-in types
   that nvcc's frontend doesn't understand. */
#ifndef _MATH_H
#  error "Never include <bits/math-vector.h> directly; include <math.h> instead."
#endif

/* Include empty stubs for __DECL_SIMD_* macros (needed by bits/mathcalls.h) */
#include <bits/libm-simd-decl-stubs.h>

/* Intentionally skip __Float32x4_t / __SVFloat32_t typedefs and vector
   math function declarations. */

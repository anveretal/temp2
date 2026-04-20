//=========================================================================
// Copyright 2023-2024 KNS Group LLC (YADRO)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//=========================================================================

#include "common.h"
#include "minigemm.h"
#include "minigemm_optimized.h"
#include "minigemm_convolution.h"
#include "direct_convolution.h"
#include "gemm_convolution.h"

#include <iostream>
#include <vector>

#include <cassert>
#include <cmath>

#ifdef INTRINSIC
#include <riscv_vector.h>
#endif

//----------------------------------------------------------------------

namespace {

// Assume data types:
// - TI (source): fp32, int32, i8/u8
// - TW (source): fp32, int32, i8/u8
// - TO (destination): fp32, int32

//----------------------------------------------------------------------

struct OptimizedConvolutionParams {
    int    IC; // input channels
    int    OC; // output channels
    int    IX; // input size by X axis (same for Y and X)
    int    S;  // stride by X and Y: S=1, or S=2
    int    K;  // kernel size, e.g: K=3 if 3x3
    Layout L;  // c_major, c_minor
    Type   T;  // fp32, int32, i8_i32, ...
    Level  O;  // optimization level
};

std::ostream& operator<<(std::ostream& out, const OptimizedConvolutionParams& params)
{
    out << "ConvolutionParams{";
    out <<   "In.C=" << params.IC;
    out << ", Out.C=" << params.OC;
    out << ", In.X=" << params.IX;
    out << ", Out.X=" << (params.IX / params.S);
    out << ", Stride="  << params.S;
    out << ", Kernel="  << params.K << "x" << params.K;
    out << ", Layout::"  << to_chars(params.L);
    out << ", Type::"  << to_chars(params.T);
    out << ", Level::"  << to_chars(params.O);
    out << "}";
    return out;
}

//----------------------------------------------------------------------

#define OPTIMIZED_CONVOLUTION_PARAMS(OptL, type)                      \
    /* Large image, few channels */                                   \
    /* IC, OC,  IX, S, K, Layout         , Type      , Level */       \
    {3, 16, 224, 1, 1, Layout::c_major, Type::type, Level::OptL},     \
    {3, 16, 224, 2, 1, Layout::c_major, Type::type, Level::OptL},     \
    {3, 16, 224, 1, 3, Layout::c_major, Type::type, Level::OptL},     \
    {3, 16, 224, 2, 3, Layout::c_major, Type::type, Level::OptL},     \
    {3, 16, 224, 1, 1, Layout::c_minor, Type::type, Level::OptL},     \
    {3, 16, 224, 2, 1, Layout::c_minor, Type::type, Level::OptL},     \
    {3, 16, 224, 1, 3, Layout::c_minor, Type::type, Level::OptL},     \
    {3, 16, 224, 2, 3, Layout::c_minor, Type::type, Level::OptL},     \
                                                                      \
    /* Medium-size image, medium channels */                          \
    /*IC,OC, IX, S, K, Layout         , Type      , Level */          \
    {64, 64, 56, 1, 1, Layout::c_major, Type::type, Level::OptL},     \
    {64, 64, 56, 2, 1, Layout::c_major, Type::type, Level::OptL},     \
    {64, 64, 56, 1, 3, Layout::c_major, Type::type, Level::OptL},     \
    {64, 64, 56, 2, 3, Layout::c_major, Type::type, Level::OptL},     \
    {64, 64, 56, 1, 1, Layout::c_minor, Type::type, Level::OptL},     \
    {64, 64, 56, 2, 1, Layout::c_minor, Type::type, Level::OptL},     \
    {64, 64, 56, 1, 3, Layout::c_minor, Type::type, Level::OptL},     \
    {64, 64, 56, 2, 3, Layout::c_minor, Type::type, Level::OptL},     \
                                                                      \
    /* Small image, med-to-many channels */                           \
    /* IC,  OC, IX, S, K, Layout         , Type      , Level */       \
    {128, 1024, 14, 1, 1, Layout::c_major, Type::type, Level::OptL},  \
    {128, 1024, 14, 2, 1, Layout::c_major, Type::type, Level::OptL},  \
    {128, 1024, 14, 1, 3, Layout::c_major, Type::type, Level::OptL},  \
    {128, 1024, 14, 2, 3, Layout::c_major, Type::type, Level::OptL},  \
    {128, 1024, 14, 1, 1, Layout::c_minor, Type::type, Level::OptL},  \
    {128, 1024, 14, 2, 1, Layout::c_minor, Type::type, Level::OptL},  \
    {128, 1024, 14, 1, 3, Layout::c_minor, Type::type, Level::OptL},  \
    {128, 1024, 14, 2, 3, Layout::c_minor, Type::type, Level::OptL},  \
                                                                      \
    /* Tiny image, many channels */                                   \
    /* IC,  OC, IX, S, K, Layout         , Type      , Level */       \
    {1024, 1024, 2, 1, 1, Layout::c_major, Type::type, Level::OptL},  \
    {1024, 1024, 2, 2, 1, Layout::c_major, Type::type, Level::OptL},  \
    {1024, 1024, 2, 1, 3, Layout::c_major, Type::type, Level::OptL},  \
    {1024, 1024, 2, 2, 3, Layout::c_major, Type::type, Level::OptL},  \
    {1024, 1024, 2, 1, 1, Layout::c_minor, Type::type, Level::OptL},  \
    {1024, 1024, 2, 2, 1, Layout::c_minor, Type::type, Level::OptL},  \
    {1024, 1024, 2, 1, 3, Layout::c_minor, Type::type, Level::OptL},  \
    {1024, 1024, 2, 2, 3, Layout::c_minor, Type::type, Level::OptL},  \

static std::vector<OptimizedConvolutionParams> test_cases = {
    //
    // Optimization Level 1: C/C++ compiler auto-vectoring for SIMD
    //
    OPTIMIZED_CONVOLUTION_PARAMS(OptL1, fp32)
    #ifdef OPTIMIZE_INTEGER
    OPTIMIZED_CONVOLUTION_PARAMS(OptL1, int32)
    OPTIMIZED_CONVOLUTION_PARAMS(OptL1, i8_i32)
    OPTIMIZED_CONVOLUTION_PARAMS(OptL1, u8_i32)
    OPTIMIZED_CONVOLUTION_PARAMS(OptL1, i8u8_i32)
    OPTIMIZED_CONVOLUTION_PARAMS(OptL1, u8i8_i32)
    #endif

#ifdef INTRINSIC

    //
    // Optimization Level 2: simple (naive) manual code vectoring for SIMD
    //                       (falls-back to level 1 if no C/C++ intrinsic)
    //
    OPTIMIZED_CONVOLUTION_PARAMS(OptL2, fp32)
    #ifdef OPTIMIZE_INTEGER
    OPTIMIZED_CONVOLUTION_PARAMS(OptL2, int32)
    OPTIMIZED_CONVOLUTION_PARAMS(OptL2, i8_i32)
    OPTIMIZED_CONVOLUTION_PARAMS(OptL2, u8_i32)
    OPTIMIZED_CONVOLUTION_PARAMS(OptL2, i8u8_i32)
    OPTIMIZED_CONVOLUTION_PARAMS(OptL2, u8i8_i32)
    #endif

    //
    // Optimization Level 3: micro-kernel, like 6x16, with SIMD registers
    //
    OPTIMIZED_CONVOLUTION_PARAMS(OptL3, fp32)
    #ifdef OPTIMIZE_INTEGER
    OPTIMIZED_CONVOLUTION_PARAMS(OptL3, int32)
    OPTIMIZED_CONVOLUTION_PARAMS(OptL3, i8_i32)
    OPTIMIZED_CONVOLUTION_PARAMS(OptL3, u8_i32)
    OPTIMIZED_CONVOLUTION_PARAMS(OptL3, i8u8_i32)
    OPTIMIZED_CONVOLUTION_PARAMS(OptL3, u8i8_i32)
    #endif

    //
    // Optimization Level 4: micro-kernel, and matrix B in L1 cache
    //
    OPTIMIZED_CONVOLUTION_PARAMS(OptL4, fp32)
    #ifdef OPTIMIZE_INTEGER
    OPTIMIZED_CONVOLUTION_PARAMS(OptL4, int32)
    OPTIMIZED_CONVOLUTION_PARAMS(OptL4, i8_i32)
    OPTIMIZED_CONVOLUTION_PARAMS(OptL4, u8_i32)
    OPTIMIZED_CONVOLUTION_PARAMS(OptL4, i8u8_i32)
    OPTIMIZED_CONVOLUTION_PARAMS(OptL4, u8i8_i32)
    #endif

    //
    // Optimization Level 5: same as Level 4, but B buffer fits L1$
    //
    OPTIMIZED_CONVOLUTION_PARAMS(OptL5, fp32)
    #ifdef OPTIMIZE_INTEGER
    OPTIMIZED_CONVOLUTION_PARAMS(OptL5, int32)
    OPTIMIZED_CONVOLUTION_PARAMS(OptL5, i8_i32)
    OPTIMIZED_CONVOLUTION_PARAMS(OptL5, u8_i32)
    OPTIMIZED_CONVOLUTION_PARAMS(OptL5, i8u8_i32)
    OPTIMIZED_CONVOLUTION_PARAMS(OptL5, u8i8_i32)
    #endif

    //
    // Optimization Level 6: same as Level 5, but also buffer A in L2$
    //
    OPTIMIZED_CONVOLUTION_PARAMS(OptL6, fp32)

#if 0 // not implemented
    //
    // Optimization Level 7: additionally cache B buffers in L3$
    //
    OPTIMIZED_CONVOLUTION_PARAMS(OptL7, fp32)
#endif // 0
#endif // INTRINSIC
};

const int test_count = test_cases.size();

//----------------------------------------------------------------------

int test_convolution(const int                         test_index,
                     const OptimizedConvolutionParams& params,
                           PerfResult                & perf_result)
{
    std::cout << "test " << (test_index + 1) << ": " << params << std::endl;

    const int  input_channels = params.IC;
    const int output_channels = params.OC;

    const int input_dim_x = params.IX;
    const int input_dim_y = input_dim_x;
    const int input_dim_c = input_channels;

    ASSERT(input_dim_x % 2 == 0, "ERROR: input size by X and Y must be even!");

    const int stride_x = params.S;
    const int stride_y = stride_x;

    ASSERT(stride_x == 1 || stride_x == 2, "ERROR: stride must equal to 1 or 2!");

    // dilations not tested
    constexpr int dilation_x = 1;
    constexpr int dilation_y = 1;

    const int output_dim_x = input_dim_x / stride_x;
    const int output_dim_y = output_dim_x;
    const int output_dim_c = output_channels;

    const int weights_dim_x = params.K;
    const int weights_dim_y = weights_dim_x;
    const int weights_dim_i = input_channels;
    const int weights_dim_o = output_channels;

    ASSERT(weights_dim_x % 2 != 0, "ERROR: kernel size must be odd!");

    // round towards zero
    const int pads_begin_x = weights_dim_x / 2;
    const int pads_begin_y = weights_dim_y / 2;

    const int pads_end_x = pads_begin_x;
    const int pads_end_y = pads_begin_y;

    //
    // How to interpret tensor axes: 0th, 1st, 2nd, 3rd
    //

    // input/output dims
    const int dim_x = 0;
    const int dim_y = 1;
    const int dim_c = 2; // channels

    // weights tensor dims
    const int dim_i = 2;
    const int dim_o = 3;

    //
    // Setup tensor dims by logical axes: x, y, channels
    //

    const int input_dims[3] {input_dim_x, input_dim_y, input_channels};
    const int output_dims[3] {output_dim_x, output_dim_y, output_channels};

    const int weights_dims[4] {weights_dim_x, weights_dim_y,
                               input_channels, output_channels};

    const int strides[2] {stride_x, stride_y};

    const int dilations[2] {dilation_x, dilation_y};

    const int pads_begin[2] {pads_begin_x, pads_begin_y};
    const int pads_end[2] {pads_end_x, pads_end_y};

    //
    // Check parameters
    //

    constexpr int spacial_ndims = 2;

    // check shapes: in, out, kernel, pads
    for (int n = 0; n < spacial_ndims; n++) {
        const int dilated_weights_dim = (weights_dims[n] - 1) * dilations[n] + 1;
        const int expected_output_dim = (input_dims[n] + pads_begin[n] + pads_end[n] - dilated_weights_dim)
                                      / strides[n] + 1;
        if (output_dims[n] != expected_output_dim) {
            std::cout << "ERROR: mismatch output_dims[" << n << "] = " << output_dims[n]
                      << " (expected" << expected_output_dim << ")"
                      << std::endl;
            return EXIT_FAILURE;
        }
    }

    //
    // Setup tensor physical steps according to layout
    // (assume dense allocation in contigious memory)
    //

    const auto layout = params.L;

    ASSERT(layout == Layout::c_major || layout == Layout::c_minor,
           "ERROR: layout must equal to c_major or c_minor!");

    int input_steps[3] {};
    int output_steps[3] {};
    int weights_steps[4] {};

    int in_out_major_dim {};
    int weights_major_dim {};

    if (layout == Layout::c_major) {
        input_steps[dim_x] = 1;
        input_steps[dim_y] = input_dim_x;
        input_steps[dim_c] = input_dim_x * input_dim_y;

        output_steps[dim_x] = 1;
        output_steps[dim_y] = output_dim_x;
        output_steps[dim_c] = output_dim_x * output_dim_y;

        weights_steps[dim_x] = 1;
        weights_steps[dim_y] = weights_dim_x;
        weights_steps[dim_i] = weights_dim_x * weights_dim_y;
        weights_steps[dim_o] = weights_dim_x * weights_dim_y * weights_dim_i;

        in_out_major_dim = dim_c;
        weights_major_dim = dim_o;
    } else { // if (layout == Layout::c_minor)
        input_steps[dim_c] = 1;
        input_steps[dim_x] = input_dim_c;
        input_steps[dim_y] = input_dim_c * input_dim_x;

        output_steps[dim_c] = 1;
        output_steps[dim_x] = output_dim_c;
        output_steps[dim_y] = output_dim_c * output_dim_x;

        weights_steps[dim_o] = 1;
        weights_steps[dim_i] = weights_dim_o;
        weights_steps[dim_y] = weights_dim_o * weights_dim_i;
        weights_steps[dim_x] = weights_dim_o * weights_dim_i * weights_dim_y;

        in_out_major_dim = dim_y;
        weights_major_dim = dim_x;
    }

    //
    // Allocate memory for tensors
    //

    const Type type = params.T;

    const int  input_length =  input_dims[in_out_major_dim] *  input_steps[in_out_major_dim];
    const int output_length = output_dims[in_out_major_dim] * output_steps[in_out_major_dim];
    const int weights_length = weights_dims[weights_major_dim] * weights_steps[weights_major_dim];

    // assume input and weights same byte-size, e.g: U8 x I8
    const int   input_element_size = in_bytes_per_elem(type);
    const int weights_element_size = input_element_size;

    // ...while output may be another byte-size, e.g: I32
    const int output_element_size = out_bytes_per_elem(type);

    const int reference_length = output_length;
    const int reference_element_size = output_element_size;

    std::vector<uint8_t> input_vector(input_length * input_element_size);
    std::vector<uint8_t> output_vector(output_length * output_element_size);
    std::vector<uint8_t> weights_vector(weights_length * weights_element_size);
    std::vector<uint8_t> reference_vector(reference_length * reference_element_size);

    uint8_t* input = input_vector.data();
    uint8_t* output = output_vector.data();
    uint8_t* weights = weights_vector.data();
    uint8_t* reference = reference_vector.data();

    //
    // Fill tensors with initial data
    //

    unsigned input_seed = 129u;
    unsigned output_seed = 137u;
    unsigned weights_seed = 145u;

    if (type == Type::fp32) {
        fill_vector<float>(input  ,   input_length, -128, 127,   input_seed);
        fill_vector<float>(weights, weights_length, -128, 127, weights_seed);
        fill_vector<float>(output ,  output_length, -128, 127,  output_seed);
        copy_vector<float>(output ,  output_length, reference);
    } else if (type == Type::int32) {
        fill_vector<int>(input  ,   input_length, -128, 127,   input_seed);
        fill_vector<int>(weights, weights_length, -128, 127, weights_seed);
        fill_vector<int>(output ,  output_length, -128, 127,  output_seed);
        copy_vector<int>(output ,  output_length, reference);
    } else if (type == Type::i8_i32) {
        fill_vector< int8_t>(input  ,   input_length, -128, 127,   input_seed);
        fill_vector< int8_t>(weights, weights_length, -128, 127, weights_seed);
        fill_vector<int32_t>(output ,  output_length, -128, 127,  output_seed);
        copy_vector<int32_t>(output ,  output_length, reference);
    } else if (type == Type::u8_i32) {
        fill_vector<uint8_t>(input  ,   input_length,    0, 255,   input_seed);
        fill_vector<uint8_t>(weights, weights_length,    0, 255, weights_seed);
        fill_vector<int32_t>(output ,  output_length, -128, 127,  output_seed);
        copy_vector<int32_t>(output ,  output_length, reference);
    } else if (type == Type::i8u8_i32) {
        fill_vector< int8_t>(input  ,   input_length, -128, 127,   input_seed);
        fill_vector<uint8_t>(weights, weights_length,    0, 255, weights_seed);
        fill_vector<int32_t>(output ,  output_length, -128, 127,  output_seed);
        copy_vector<int32_t>(output ,  output_length, reference);
    } else if (type == Type::u8i8_i32) {
        fill_vector<uint8_t>(input  ,   input_length,    0, 255,   input_seed);
        fill_vector< int8_t>(weights, weights_length, -128, 127, weights_seed);
        fill_vector<int32_t>(output ,  output_length, -128, 127,  output_seed);
        copy_vector<int32_t>(output ,  output_length, reference);
#if 0
//
// Optimization for types other that FP32 and I32 is not implemented yet
//
#ifdef WITH_FP16
    } else if (type == Type::fp16) {
        fill_vector<fp16_t>(input  ,   input_length,  11.2, 14.4,   input_seed);
        fill_vector<fp16_t>(weights, weights_length, -12.8, 12.7, weights_seed);
        fill_vector<fp16_t>(output ,  output_length, -12.8, 12.7,  output_seed);
        copy_vector<fp16_t>(output ,  output_length, reference);
    } else if (type == Type::f16_f32) {
        fill_vector<fp16_t>(input  ,   input_length,    0, 255,   input_seed);
        fill_vector<fp16_t>(weights, weights_length, -128, 127, weights_seed);
        fill_vector< float>(output ,  output_length, -128, 127,  output_seed);
        copy_vector< float>(output ,  output_length, reference);
#endif // FP16
#endif // 0
    } else {
        ASSERT(false, "ERROR: unsupported type!");
    }

    //
    // Define callers for reference and main convolution functions
    //

    const Level level = params.O; // mini-gemm optimization level

    #ifdef WITH_OPEN_BLAS
        // tensor length as measured by element items
        // note: input will be duplicated by im2col transform
        const int  input_dense_length =  output_dim_x *  output_dim_y
                                      * weights_dim_x * weights_dim_y
                                      * input_dim_c;
        const int output_dense_length = output_dim_x * output_dim_y
                                      * output_dim_c;
        const int weights_dense_length = weights_dim_x * weights_dim_y
                                       * weights_dim_i * weights_dim_o;
        const int buffer_length =  input_dense_length
                                + output_dense_length
                                + weights_dense_length;
        std::vector<float> buffer_vector(buffer_length);
        float* buffer = buffer_vector.data();
    #else
        const int buffer_length = 0;
        float* buffer = nullptr;
    #endif

    #define CALL_GEMM_CONVOLUTION(TI, TW, TO)                           \
        status = gemm_convolution(reinterpret_cast<const TI*>(input),   \
                                  reinterpret_cast<const TW*>(weights), \
                                  reinterpret_cast<TO*>(reference),     \
                                     input_dims, input_steps,           \
                                     weights_dims, weights_steps,       \
                                     output_dims, output_steps,         \
                                     strides, dilations,                \
                                     pads_begin, pads_end,              \
                                     buffer, buffer_length);            \
        return status;

    const auto call_gemm_convolution = [&]() {
        int status = EXIT_SUCCESS;
        if (type == Type::fp32) {
            CALL_GEMM_CONVOLUTION(float, float, float)
        } else if (type == Type::int32) {
            CALL_GEMM_CONVOLUTION(int32_t, int32_t, int32_t)
        } else if (type == Type::i8_i32) {
            CALL_GEMM_CONVOLUTION(int8_t,  int8_t, int32_t)
        } else if (type == Type::u8_i32) {
            CALL_GEMM_CONVOLUTION(uint8_t, uint8_t, int32_t)
        } else if (type == Type::i8u8_i32) {
            CALL_GEMM_CONVOLUTION(int8_t, uint8_t, int32_t)
        } else if (type == Type::u8i8_i32) {
            CALL_GEMM_CONVOLUTION(uint8_t,  int8_t, int32_t)
    #if 0 // not implemented
    #ifdef WITH_FP16
        } else if (type == Type::fp16) {
            CALL_GEMM_CONVOLUTION(fp16_t, fp16_t, fp16_t)
        } else if (type == Type::f16_f32) {
            CALL_GEMM_CONVOLUTION(fp16_t, fp16_t, float)
    #endif // FP16
    #endif // 0
        } else {
            ASSERT(false, "ERROR: unsupported type!");
        }
        return status;
    };

    #undef CALL_GEMM_CONVOLUTION

    #define CALL_CONVOLUTION(PREFIX, TI, TW, TO, OUT, EXTRA)               \
        status = PREFIX##convolution(reinterpret_cast<const TI*>(input),   \
                                     reinterpret_cast<const TW*>(weights), \
                                     reinterpret_cast<TO*>(OUT),           \
                                     input_dims, input_steps,              \
                                     weights_dims, weights_steps,          \
                                     output_dims, output_steps,            \
                                     strides, dilations,                   \
                                     pads_begin, pads_end,                 \
                                     EXTRA);                               \
        return status;

    const auto call_direct_convolution = [&]() {
        int status = EXIT_SUCCESS;
        if (type == Type::fp32) {
            CALL_CONVOLUTION(direct_, float, float, float, reference, layout)
        } else if (type == Type::int32) {
            CALL_CONVOLUTION(direct_, int32_t, int32_t, int32_t, reference, layout)
        } else if (type == Type::i8_i32) {
            CALL_CONVOLUTION(direct_,  int8_t,  int8_t, int32_t, reference, layout)
        } else if (type == Type::u8_i32) {
            CALL_CONVOLUTION(direct_, uint8_t, uint8_t, int32_t, reference, layout)
        } else if (type == Type::i8u8_i32) {
            CALL_CONVOLUTION(direct_,  int8_t, uint8_t, int32_t, reference, layout)
        } else if (type == Type::u8i8_i32) {
            CALL_CONVOLUTION(direct_, uint8_t,  int8_t, int32_t, reference, layout)
    #if 0 // not implemented
    #ifdef WITH_FP16
        } else if (type == Type::fp16) {
            CALL_CONVOLUTION(direct_, fp16_t, fp16_t, fp16_t, reference, layout)
        } else if (type == Type::f16_f32) {
            CALL_CONVOLUTION(direct_, fp16_t, fp16_t,  float, reference, layout)
    #endif // FP16
    #endif // 0
        } else {
            ASSERT(false, "ERROR: unsupported type!");
        }
        return status;
    };

    const auto call_minigemm_convolution = [&]() {
        int status = EXIT_SUCCESS;
        if (type == Type::fp32) {
            CALL_CONVOLUTION(minigemm_, float, float, float, output, level)
        } else if (type == Type::int32) {
            CALL_CONVOLUTION(minigemm_, int32_t, int32_t, int32_t, output, level)
        } else if (type == Type::i8_i32) {
            CALL_CONVOLUTION(minigemm_,  int8_t,  int8_t, int32_t, output, level)
        } else if (type == Type::u8_i32) {
            CALL_CONVOLUTION(minigemm_, uint8_t, uint8_t, int32_t, output, level)
        } else if (type == Type::i8u8_i32) {
            CALL_CONVOLUTION(minigemm_,  int8_t, uint8_t, int32_t, output, level)
        } else if (type == Type::u8i8_i32) {
            CALL_CONVOLUTION(minigemm_, uint8_t,  int8_t, int32_t, output, level)
    #if 0 // not implemented
    #ifdef WITH_FP16
        } else if (type == Type::fp16) {
            CALL_CONVOLUTION(minigemm_, fp16_t, fp16_t, fp16_t, output, level)
        } else if (type == Type::f16_f32) {
            CALL_CONVOLUTION(minigemm_, fp16_t, fp16_t,  float, output, level)
    #endif // FP16
    #endif // 0
        } else {
            ASSERT(false, "ERROR: unsupported type!");
        }
        return status;
    };

    #undef CALL_CONVOLUTION

    //
    // Performance testing:
    //

    #ifdef WITH_OPEN_BLAS
        const auto& call_reference = call_gemm_convolution;
        (void) call_direct_convolution; // unused
    #else
        const auto& call_reference = call_direct_convolution;
        (void) call_gemm_convolution; // unused
    #endif

    const auto& call_convolution = call_minigemm_convolution;

    const float ref_ms = perf_test(call_reference);
    std::cout << "  ref (ms): " << ref_ms << std::endl;

    const float opt_ms = perf_test(call_convolution);
    std::cout << "  opt (ms): " << opt_ms << std::endl;

    const float speedup = ref_ms / opt_ms;
    std::cout << "  speedup: " << speedup << " times" << std::endl;

    perf_result.ref_ms = ref_ms;
    perf_result.opt_ms = opt_ms;

    //
    // Result correctness check:
    //

    int status = EXIT_SUCCESS;

    if (type == Type::fp32) {
        status = check_vector<float>(output, output_length, reference);
    } else if (type == Type::int32 ||
               type == Type::i8_i32 ||
               type == Type::u8_i32 ||
               type == Type::i8u8_i32 ||
               type == Type::u8i8_i32) {
        status = check_vector<int>(output, output_length, reference);
#if 0 // not implemented
#ifdef WITH_FP16
    } else if (type == Type::fp16) {
        status = check_vector<fp16_t>(output, output_length, reference);
    } else if (type == Type::f16_f32) {
        status = check_vector<float>(output, output_length, reference);
#endif // FP16
#endif // 0
    } else {
        ASSERT(false, "ERROR: unsupported type!");
    }

    if (status != EXIT_SUCCESS) {
        std::cout << "EXIT_FAILURE" << std::endl;
    }

    return status;
}

//----------------------------------------------------------------------

// summarize performance results like a table for Excel
void print_perf_results(const PerfResult                 perf_results[],
                        const OptimizedConvolutionParams test_cases[],
                        const int                        test_count)
{
    std::cout << std::endl;

    // Header of table
    std::cout << "#\tIn.C\tOut.C\tIn.X\tOut.X\tStride\tKernel\tLayout\tType\tLevel"
              << "\tRef(ms)\tOpt(ms)"
              << std::endl;

    for (int i = 0; i < test_count; i++) {
        const auto& t = test_cases[i];
        const auto& r = perf_results[i];
        std::cout << (i + 1)
                  << "\t" << t.IC
                  << "\t" << t.OC
                  << "\t" << t.IX
                  << "\t" << (t.IX / t.S)
                  << "\t" << t.S
                  << "\t" << t.K << "x" << t.K
                  << "\t" << to_chars(t.L)
                  << "\t" << to_chars(t.T)
                  << "\t" << to_chars(t.O)
                  << "\t" << r.ref_ms
                  << "\t" << r.opt_ms
                  << std::endl;
    }

    std::cout << std::endl;
}

//----------------------------------------------------------------------

int example_convolution() {
    std::cout << "example: optimized convolution" << std::endl;

    std::vector<PerfResult> perf_results_vector(test_count);
    PerfResult* perf_results = perf_results_vector.data();

    for (int i = 0; i < test_count; i++) {
        const auto& t = test_cases[i];
              auto& r = perf_results[i];
        int status = test_convolution(i, t, r);
        if (status != EXIT_SUCCESS) {
            return EXIT_FAILURE;
        }
    }

    const auto* test_cases_ptr = test_cases.data();
    print_perf_results(perf_results, test_cases_ptr, test_count);

    return EXIT_SUCCESS;
}

} // anonymous namespace

//----------------------------------------------------------------------

int main(const int argc, const char *argv[]) {
    int status = exec_main(argc, argv, example_convolution);
    return status;
}

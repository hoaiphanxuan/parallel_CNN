#ifndef GPU_FORWARD_H
#define GPU_FORWARD_H
#include "./gpu_utils.h"

class Forward_GPU
{
    public:
    //basic
    static void forward_GPU_basic(float *output, float *input, float *weight, float *bias,
    int n_sample, int channel_out, int channel_in,
    int height_in, int width_in, int kernel_height);

    //Unrolling loop
    static void forward_GPU_optimize_0(float *output, float *input, float *weight, float *bias,
    int n_sample, int channel_out, int channel_in,
    int height_in, int width_in, int kernel_height);

    //use shared memory and constant memory
    static void forward_GPU_optimize_1(float *output, float *input, float *weight, float *bias,
    int n_sample, int channel_out, int channel_in,
    int height_in, int width_in, int kernel_height);

    //use stream 
    static void forward_GPU_optimize_2(float *output, float *input, float *weight, float *bias,
    int n_sample, int channel_out, int channel_in,
    int height_in, int width_in, int kernel_height);

    //use shared memory, constant memory, stream, and loop unrolling
    static void forward_GPU_optimize_3(float *output, float *input, float *weight, float *bias,
    int n_sample, int channel_out, int channel_in,
    int height_in, int width_in, int kernel_height);

};

#endif
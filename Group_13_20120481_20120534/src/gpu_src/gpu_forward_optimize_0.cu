#include "./gpu_forward.h"
#include <cmath>
#include <iostream>

#define BLOCKSIZE 16


__global__ void conv_optimize_kernel_0(float *output, float *input, float *weight, float *bias, 
    int channel_out, int channel_in, int height_in, int width_in, int kernel_size)
{
    
    int height_out = height_in - kernel_size + 1;
    int width_out = width_in - kernel_size + 1;

    int sample_idx = blockIdx.x;        // sample_idx
    int channel_out_idx = blockIdx.y; //out_chanel_idx

    // index của thread
    int index_per_block = blockIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

    //  Tính vị trí index của ảnh đầu ra
    int r = index_per_block / width_out; // row of the output image matrix
    int c = index_per_block % width_out; // col of the output image matrix

    float sum = 0.0f;
    int half_kernel_size = kernel_size / 2;
    int index_per_kernel = 0;

    int r_input = r + half_kernel_size;
    int c_input = c + half_kernel_size;

    if (r < height_out && c < width_out) 
    {
        #pragma unroll
        for(int k_row = -half_kernel_size; k_row <= half_kernel_size; k_row++)          
        {
            #pragma unroll
            for(int k_col = -half_kernel_size; k_col <= half_kernel_size; k_col++)
            {
                int row = r_input + k_row;
                int col = c_input + k_col;
                
                #pragma unroll
                for(int channel_in_idx = 0; channel_in_idx < channel_in; channel_in_idx++)             
                {
                    sum += input[(sample_idx * (channel_in * height_in * width_in)) + 
                        (channel_in_idx * (height_in * width_in)) + 
                        (row * width_in) + col] *
                    weight[(channel_out_idx * (channel_in * kernel_size * kernel_size)) + 
                            (channel_in_idx * (kernel_size * kernel_size)) + index_per_kernel];
                }
                index_per_kernel++;

            }
        }
        output[(sample_idx * (channel_out * height_out * width_out)) + 
        (channel_out_idx * (height_out * width_out)) + (r * width_out) + c] = sum + bias[channel_out_idx];
    }
}

void Forward_GPU::forward_GPU_optimize_0(float *output, float *input, float *weight, float * bias,
    int n_samples, int channel_out, int channel_in,
    int height_in, int width_in, int kernel_height)
{
    std:: cout << "optimize 0: Unrolling loop" << std :: endl ;
    int height_out = height_in - kernel_height + 1;
    int width_out = width_in - kernel_height + 1;

    // Allocate memory
    float *d_input, *d_ouput, *d_weight, *d_bias;
    CHECK(cudaMalloc((void **)&d_input, n_samples * channel_in * height_in * width_in * sizeof(float)));  
    CHECK(cudaMalloc((void **)&d_ouput, n_samples * channel_out * height_out * width_out * sizeof(float)));  
    CHECK(cudaMalloc((void **)&d_weight, channel_out * channel_in * kernel_height * kernel_height * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_bias, channel_out * sizeof(float)));  

    GpuTimer timer;
    timer.Start();

    // Copy data to device
    CHECK(cudaMemcpy(d_input, input, n_samples * channel_in * height_in * width_in * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_weight, weight, channel_out * channel_in * kernel_height * kernel_height * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_bias, bias, channel_out * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);
    dim3 gridSize(n_samples, channel_out, (height_out * width_out - 1) / (BLOCKSIZE * BLOCKSIZE) + 1);

    // Call the kernel
    conv_optimize_kernel_0<<<gridSize, blockSize>>>(d_ouput, d_input, d_weight, d_bias, channel_out, channel_in, height_in, width_in, kernel_height);
    CHECK(cudaGetLastError());

    // Copy data to host
    CHECK(cudaMemcpy(output, d_ouput, n_samples * channel_out * height_out * width_out * sizeof(float), cudaMemcpyDeviceToHost));

    timer.Stop();
	float time = timer.Elapsed();

    int layer = 1;
    if (channel_in != 1)
      layer = 3;

    std::cout << "  - Kernel time C" << layer <<" : " << time << " ms" << std::endl;

    // Free device memory
    
    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_ouput));
    CHECK(cudaFree(d_weight));
    CHECK(cudaFree(d_bias));
}


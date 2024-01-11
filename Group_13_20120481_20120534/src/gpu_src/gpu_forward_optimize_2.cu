#include "./gpu_forward.h"
#include <cmath>
#include <iostream>

#define BLOCKSIZE 16

__global__ void conv_optimize_kernel_2(float *output, float *input, float *kernel, float *bias,
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
        
        for(int k_row = -half_kernel_size; k_row <= half_kernel_size; k_row++)          
        {
            
            for(int k_col = -half_kernel_size; k_col <= half_kernel_size; k_col++)
            {
                int row = r_input + k_row;
                int col = c_input + k_col;
                
                for(int channel_in_idx = 0; channel_in_idx < channel_in; channel_in_idx++)             
                {
                    sum += input[(sample_idx * (channel_in * height_in * width_in)) + 
                        (channel_in_idx * (height_in * width_in)) + 
                        (row * width_in) + col] *
                    kernel[(channel_out_idx * (channel_in * kernel_size * kernel_size)) + 
                            (channel_in_idx * (kernel_size * kernel_size)) + index_per_kernel] ;
                }
                index_per_kernel++;

            }
        }
        output[(sample_idx * (channel_out * height_out * width_out)) + 
        (channel_out_idx * (height_out * width_out)) + (r * width_out) + c] = sum + bias[channel_out_idx];
    }
}

void Forward_GPU::forward_GPU_optimize_2(float *output, float *input, float *weight, float * bias,
    int n_samples, int channel_out, int channel_in,
    int height_in, int width_in, int kernel_height)
{
    std:: cout << "optimize 2: stream"<< std::endl;

    int height_out = height_in - kernel_height + 1;
    int width_out = width_in - kernel_height + 1;

    int nStreams = 10;
    cudaStream_t stream[const_cast<const int&>(nStreams)];

    for (int i = 0; i < nStreams; i++)
    {
      CHECK(cudaStreamCreate(&stream[i]));
    }

    size_t input_per_stream = channel_in * height_in *  width_in ;
    size_t output_per_stream = channel_out * height_out *  width_out;

		// Allocate device memory
    float *d_input, *d_ouput, *d_weight, *d_bias;
    CHECK(cudaMalloc((void **)&d_input, n_samples * channel_in * height_in * width_in * sizeof(float)));  
    CHECK(cudaMalloc((void **)&d_ouput, n_samples * channel_out * height_out * width_out * sizeof(float)));  
    CHECK(cudaMalloc((void **)&d_weight, channel_out * channel_in * kernel_height * kernel_height * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_bias, channel_out * sizeof(float)));

    int len_per_part = n_samples / nStreams;

    GpuTimer timer;
    timer.Start();

    CHECK(cudaMemcpy(d_weight, weight, channel_out * channel_in * kernel_height * kernel_height * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_bias, bias, channel_out * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);
    dim3 gridSize(len_per_part, channel_out, (height_out * width_out - 1) / (BLOCKSIZE * BLOCKSIZE) + 1);

    for(int i = 0; i < nStreams;i++)
    {
      int in_start = i * input_per_stream * len_per_part;
      int out_start = i * output_per_stream * len_per_part;

      CHECK(cudaMemcpyAsync(&d_input[in_start],&input[in_start], len_per_part * input_per_stream * sizeof(float),cudaMemcpyHostToDevice, stream[i]));

      conv_optimize_kernel_2<<<gridSize, blockSize,channel_in * height_in * width_in * sizeof(float),stream[i]>>>(&d_ouput[out_start], &d_input[in_start], d_weight, d_bias, channel_out, channel_in, height_in, width_in, kernel_height);
      CHECK(cudaGetLastError());

      CHECK(cudaMemcpyAsync(&output[out_start],&d_ouput[out_start],len_per_part * output_per_stream * sizeof(float),cudaMemcpyDeviceToHost, stream[i]));
    }

    timer.Stop();
	float time = timer.Elapsed();

    int layer = 1;
    if (channel_in != 1)
      layer = 3;

    std::cout << "  - Kernel time C" << layer <<" : " << time << " ms" << std::endl;

    // Free device memory

    for (int i = 0; i < nStreams; i++)
    {
      CHECK(cudaStreamSynchronize(stream[i]));
      CHECK(cudaStreamDestroy(stream[i]));
    }

    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_ouput));
    CHECK(cudaFree(d_weight));
    CHECK(cudaFree(d_bias));
}


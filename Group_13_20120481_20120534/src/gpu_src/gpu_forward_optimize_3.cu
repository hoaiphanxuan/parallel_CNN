#include "./gpu_forward.h"
#include <cmath>
#include <iostream>

__constant__ float dc_filter[2400]; // 16 x 5 x 5 x 6
__constant__ float dc_bias[16];

#define BLOCKSIZE 8

__global__ void conv_optimize_kernel_3(float *output, float *input,
    int channel_out, int channel_in, int height_in, int width_in, int kernel_size)
{

  extern __shared__ float s_input[];

  int sample_idx = blockIdx.x;        // sample_idx

  int num_thread_per_block = blockDim.x * blockDim.y * blockDim.z;

  int share_size = width_in * height_in * channel_in;

  int num_loop = share_size / num_thread_per_block;

  int index_per_block = blockIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

  for (int i = 0; i <= num_loop; i++)
  {
    int share_index =  index_per_block + i * num_thread_per_block;
    int input_index = share_index;
    if(share_index < share_size)
    {
    	s_input[share_index] = input[sample_idx * share_size + input_index];
    }	
    		
  }
  __syncthreads(); 

  int height_out = height_in - kernel_size + 1;
  int width_out = width_in - kernel_size + 1;

  
  int channel_out_idx = blockIdx.y; //out_chanel_idx

  //  Tính vị trí index của ảnh đầu ra
  int r = index_per_block / width_out; 
  int c = index_per_block % width_out; 

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
                    sum += s_input[ channel_in_idx * height_in * width_in + (row * width_in) + col] *
                    dc_filter[(channel_out_idx * (channel_in * kernel_size * kernel_size)) + 
                            (channel_in_idx * (kernel_size * kernel_size)) + index_per_kernel];
                }
                index_per_kernel++;

            }
        }
        output[(sample_idx * (channel_out * height_out * width_out)) + 
        (channel_out_idx * (height_out * width_out)) + (r * width_out) + c] = sum + dc_bias[channel_out_idx];
  }

}

void Forward_GPU::forward_GPU_optimize_3(float *output, float *input, float *weight, float *bias,
    int n_samples, int channel_out, int channel_in,
    int height_in, int width_in, int kernel_height)
{
    std:: cout << "optimize 3: stream, shared memory, constant memory, and unrolling loop" << std::endl;

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
    float *d_input, *d_ouput, *d_weight;
    CHECK(cudaMalloc((void **)&d_input, n_samples * channel_in * height_in * width_in * sizeof(float)));  
    CHECK(cudaMalloc((void **)&d_ouput, n_samples * channel_out * height_out * width_out * sizeof(float)));  
    CHECK(cudaMalloc((void **)&d_weight, channel_out * channel_in * kernel_height * kernel_height * sizeof(float)));

    int len_per_part = n_samples / nStreams;

    GpuTimer timer;
    timer.Start();

    CHECK(cudaMemcpyToSymbol(dc_filter, weight,channel_in * kernel_height * kernel_height * channel_out * sizeof(float)));
    CHECK(cudaMemcpyToSymbol(dc_bias, bias,channel_out * sizeof(float)));

    dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);
    dim3 gridSize(len_per_part, channel_out, (height_out * width_out - 1) / (BLOCKSIZE * BLOCKSIZE) + 1);

    for(int i = 0; i < nStreams;i++)
    {
      int in_start = i * input_per_stream * len_per_part;
      int out_start = i * output_per_stream * len_per_part;

      CHECK(cudaMemcpyAsync(&d_input[in_start],&input[in_start], len_per_part * input_per_stream * sizeof(float),cudaMemcpyHostToDevice, stream[i]));

      conv_optimize_kernel_3<<<gridSize, blockSize,channel_in * height_in * width_in * sizeof(float),stream[i]>>>(&d_ouput[out_start], &d_input[in_start], channel_out, channel_in, height_in, width_in, kernel_height);
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
}


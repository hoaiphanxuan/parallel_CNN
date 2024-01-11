#include "./gpu_forward.h"
#include <cmath>
#include <iostream>

__constant__ float dc_filter[2400]; // 16 x 5 x 5 x 6
__constant__ float dc_bias[16];

#define BLOCKSIZE 16

__global__ void conv_optimize_kernel_1(float *output, float *input,
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
        for(int k_row = -half_kernel_size; k_row <= half_kernel_size; k_row++)          
        {
            for(int k_col = -half_kernel_size; k_col <= half_kernel_size; k_col++)
            {
                int row = r_input + k_row;
                int col = c_input + k_col;
                
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

void Forward_GPU::forward_GPU_optimize_1(float *output, float *input, float *weight, float *bias,
    int n_samples, int channel_out, int channel_in,
    int height_in, int width_in, int kernel_height)
{
    std:: cout << "optimize 1: shared memory and constant memory" << std :: endl;

    int height_out = height_in - kernel_height + 1;
    int width_out = width_in - kernel_height + 1;

    // Allocate memory
    float *d_input, *d_ouput, *d_weight;
    CHECK(cudaMalloc((void **)&d_input, n_samples * channel_in * height_in * width_in * sizeof(float)));  
    CHECK(cudaMalloc((void **)&d_ouput, n_samples * channel_out * height_out * width_out * sizeof(float)));  
    CHECK(cudaMalloc((void **)&d_weight, channel_out * channel_in * kernel_height * kernel_height * sizeof(float)));  

    GpuTimer timer;
    timer.Start();

    // Copy data to device
    CHECK(cudaMemcpy(d_input, input, n_samples * channel_in * height_in * width_in * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(dc_filter, weight,channel_in * kernel_height * kernel_height * channel_out * sizeof(float)));
    CHECK(cudaMemcpyToSymbol(dc_bias, bias,channel_out * sizeof(float)));

    dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);
    dim3 gridSize(n_samples, channel_out, (height_out * width_out - 1) / (BLOCKSIZE * BLOCKSIZE) + 1);


    // Call the kernel
    conv_optimize_kernel_1<<<gridSize, blockSize,channel_in * height_in * width_in * sizeof(float)>>>(d_ouput, d_input, channel_out, channel_in, height_in, width_in, kernel_height);
    CHECK(cudaGetLastError());

    // Copy data to host
    CHECK(cudaMemcpy(output, d_ouput, n_samples * channel_out * height_out * width_out * sizeof(float), cudaMemcpyDeviceToHost));

    timer.Stop();
	float time = timer.Elapsed();

    int layer = 1;
    if (channel_in != 1)
      layer = 3;

    std::cout << "  - Kernel time C" << layer <<" : " << time << " ms" << std::endl;

    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_ouput));
    CHECK(cudaFree(d_weight));
}


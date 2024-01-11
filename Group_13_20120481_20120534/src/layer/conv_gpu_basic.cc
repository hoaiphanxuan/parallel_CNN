#include "./conv_gpu_basic.h"
#include <math.h>
#include <iostream>


void Conv_GPU_Basic::init()
{
    height_out = (1 + (height_in - height_kernel + 2 * pad_h) / stride);
    width_out = (1 + (width_in - width_kernel + 2 * pad_w) / stride);
    dim_out = height_out * width_out * channel_out;

    weight.resize(channel_in * height_kernel * width_kernel, channel_out);
    bias.resize(channel_out);

    set_normal_random(weight.data(), weight.size(), 0, 0.01);
    set_normal_random(bias.data(), bias.size(), 0, 0.01);

}

void Conv_GPU_Basic::forward(const Matrix &bottom)
{   
    int n_sample = bottom.cols();
    top.resize(height_out * width_out * channel_out, n_sample);

    float *input = (float *)bottom.data();
    float *output = (float *)top.data();
    float *w = (float *)weight.data();
    float *b = (float *)bias.data();
    
    // cudaHostRegister(input, bottom.size()* sizeof(float), cudaHostRegisterDefault);
    // cudaHostRegister(output, top.size()* sizeof(float), cudaHostRegisterDefault);

    GpuTimer timer;
	timer.Start();

    
    Forward_GPU::forward_GPU_basic(output, input, w, b,
                                    n_sample, channel_out, channel_in,
                                    height_in, width_in, height_kernel);

    timer.Stop();
	  float time = timer.Elapsed();

    int layer = 1;
    if (channel_in != 1)
      layer = 3;
    
    std::cout << "  - Convolution C" << layer <<": " << time << " ms" << std::endl << std::endl;
 
}

std::vector<float> Conv_GPU_Basic::get_parameters() const
{
    std::vector<float> res(weight.size() + bias.size());
    // Copy the data of weights and bias to a long vector
    std::copy(weight.data(), weight.data() + weight.size(), res.begin());
    std::copy(bias.data(), bias.data() + bias.size(), res.begin() + weight.size());
    return res;
}

void Conv_GPU_Basic::set_parameters(const std::vector<float> &param)
{
    if (static_cast<int>(param.size()) != weight.size() + bias.size())
        throw std::invalid_argument("Parameter size does not match");
    std::copy(param.begin(), param.begin() + weight.size(), weight.data());
    std::copy(param.begin() + weight.size(), param.end(), bias.data());
}

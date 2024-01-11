#include <Eigen/Dense>
#include <algorithm>
#include <iostream>

#include "src/layer.h"
#include "src/layer/conv_gpu_optimize_1.h"
#include "src/layer/fully_connected.h"
#include "src/layer/max_pooling.h"
#include "src/layer/relu.h"
#include "src/layer/softmax.h"
#include "src/fashion_mnist.h"
#include "src/network.h"
#include "src/gpu_src/gpu_utils.h"


int main()
{
    printDeviceInfo();    
    
    //Load data
    FASHION_MNIST dataset("./data/fashion/");
    dataset.read();

    std::cout << "number test samples: " << dataset.test_labels.cols() << std::endl;
    
    float accuracy = 0.0;

    Network lanet_5;
    Layer *conv1 = new Conv_GPU_Optimize_1(1, 28, 28, 6, 5, 5);
    Layer *pool1 = new MaxPooling(6, 24, 24, 2, 2, 2);
    Layer *conv2 = new Conv_GPU_Optimize_1(6, 12, 12, 16, 5, 5);
    Layer *pool2 = new MaxPooling(16, 8, 8, 2, 2, 2);
    Layer *fc1 = new FullyConnected(pool2->output_dim(), 120);
    Layer *fc2 = new FullyConnected(120, 84);
    Layer *fc3 = new FullyConnected(84, 10);
    Layer *relu1 = new ReLU;
    Layer *relu2 = new ReLU;
    Layer *relu3 = new ReLU;
    Layer *relu4 = new ReLU;
    Layer *softmax = new Softmax;
    lanet_5.add_layer(conv1);
    lanet_5.add_layer(relu1);
    lanet_5.add_layer(pool1);
    lanet_5.add_layer(conv2);
    lanet_5.add_layer(relu2);
    lanet_5.add_layer(pool2);
    lanet_5.add_layer(fc1);
    lanet_5.add_layer(relu3);
    lanet_5.add_layer(fc2);
    lanet_5.add_layer(relu4);
    lanet_5.add_layer(fc3);
    lanet_5.add_layer(softmax);
    lanet_5.load_model("./model/lanet_5.bin");
    lanet_5.forward(dataset.test_data);
    accuracy = compute_accuracy(lanet_5.output(), dataset.test_labels);
    std::cout << "test accuracy: " << accuracy << std::endl;
    
    return 0;
}
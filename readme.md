
<div style="text-align: center">
    <span style="font-size: 3em; font-weight: 700; font-family: Consolas">
        PARALLEL PROGRAMMING <br>
    </span>
    <span style="font-size: 3em; font-weight: 200; font-family: Consolas">
        FINAL PROJECT <br>
    </span>
    <br><br>
    <span style="font-size: 2em">
       Parallelizing and optimizing the forward-pass of a 
convolutional layer using CUDA
    </span>
</div>


## Collaborators 
- `20120481` **Phan Xuân Hoài** (20120481@student.hcmus.edu.vn)
- `20120534` **Nguyễn Minh Nghĩa** (20120534@student.hcmus.edu.vn)
## Instructors
- `HCMUS` thầy **Phạm Trọng Nghĩa** ([@ptnghia](ptnghia@fit.hcmus.edu.vn))
---
<div style="page-break-after: always"></div>

## Description
> In this final project, we will be implementing and optimizing the forward-pass of a convolutional layer using CUDA. Convolutional layers are the primary building blocks 
of convolutional neural networks (CNNs), which are used in many machine learning tasks like image classification,  object detection, natural language processing, and  recommendation systems. In general, CNNs work well on tasks where the data/input features have some level of spatial relationship.

> We optimized CUDA implementation of the convolutional layer will be used to perform inference for layers C1 and C3. 

> We will use the Fashion MNIST dataset, where the inputs to the network will be a single channel images with dimensions of 28 x 28 pixels. The output layer consists of 
10 nodes, where each node represents the likelihood of the input belonging to one of the 10 classes (T-shirt, dress, sneaker, boot etc.)

> The overall learning objectives for this project are:
*Demonstrating command of CUDA and optimization approaches by designing and implementing an optimized neural-network convolutional layer forward pass.*
---
## How to run project?
To run project, we need to download the project at [Drive](https://drive.google.com/file/d/1gHyKXiVSq8PlU75mK5bYbIZELNHyQZxh/view?usp=sharing).

First, we need to unzip and go into the folder

> !unzip -q 20120481_20120534.zip

> %cd 20120481_20120534

Then, we just go ahead to run the code with the following command to start project: 

> make compile

- To run code with serial convolution:

>make -s host_convolution

- To run code with basic kernel parallel convolution:

>make -s kernel_convolution_basic

- To run code with optimize (version 0) kernel parallel convolution: unrolling loop

>make -s kernel_convolution_optimize_0

- To run code with optimize (version 1) kernel parallel convolution: shared memory and constant memory

>make -s kernel_convolution_optimize_1

- To run code with optimize (version 2) kernel parallel convolution: CUDA stream

>make -s kernel_convolution_optimize_2

- To run code with optimize (version 3) kernel parallel convolution: CUDA stream, shared memory, constant memory, unrolling loop

>make -s kernel_convolution_optimize_3

---

## Video demo 
[demo](https://www.youtube.com/watch?v=HLLNnBKUuLM)

## References

* Standford [Convolutional Neural Networks cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks#overview)
  
* [How Convolutional Neural Networks work](https://www.youtube.com/watch?v=FmpDIaiMIeA)

* [Convolutional Neural Networks](https://d2l.ai/chapter_convolutional-neural-networks/index.html)

* LeNet-5 architecture: [Lenet-5](https://www.kaggle.com/code/blurredmachine/lenet-architecture-a-complete-guide/notebook)
 




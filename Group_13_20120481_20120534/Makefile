
cuda_lib := -L/usr/local/cuda/lib64

#---------------------------------------------------------------------------------------------------------------------------

compile:
	rm -f *.o src/*.o src/layer/*.o src/loss/*.o src/optimizer/*.o src/gpu_src/*.o
	make network.o
	make fashion_mnist.o
	make layer.o
	make loss.o
	make optimizer.o
	make basic.o
	make optimize_0.o
	make optimize_1.o
	make optimize_2.o
	make optimize_3.o
	make gpu.o

#------------------------------------------------------------------------------------------------------------------------------

network.o:
	nvcc --compile src/network.cc -o src/network.o -I./ $(cuda_lib) -lcudart

fashion_mnist.o:
	nvcc --compile src/fashion_mnist.cc -o src/fashion_mnist.o  -I./ $(cuda_lib) -lcudart

layer.o: 
	nvcc --compile src/layer/ave_pooling.cc -o src/layer/ave_pooling.o -I./ $(cuda_lib) -lcudart
	nvcc --compile src/layer/fully_connected.cc -o src/layer/fully_connected.o -I./ $(cuda_lib) -lcudart
	nvcc --compile src/layer/max_pooling.cc -o src/layer/max_pooling.o -I./ $(cuda_lib) -lcudart
	nvcc --compile src/layer/relu.cc -o src/layer/relu.o -I./ $(cuda_lib) -lcudart
	nvcc --compile src/layer/sigmoid.cc -o src/layer/sigmoid.o -I./ $(cuda_lib) -lcudart
	nvcc --compile src/layer/softmax.cc -o src/layer/softmax.o -I./ $(cuda_lib) -lcudart

	
	nvcc --compile src/layer/conv.cc -o src/layer/conv.o -I./ $(cuda_lib) -lcudart 
	nvcc --compile src/layer/conv_gpu_basic.cc -o src/layer/conv_gpu_basic.o -I./ $(cuda_lib) -lcudart
	nvcc --compile src/layer/conv_gpu_optimize_0.cc -o src/layer/conv_gpu_optimize_0.o -I./ $(cuda_lib) -lcudart
	nvcc --compile src/layer/conv_gpu_optimize_1.cc -o src/layer/conv_gpu_optimize_1.o -I./ $(cuda_lib) -lcudart
	nvcc --compile src/layer/conv_gpu_optimize_2.cc -o src/layer/conv_gpu_optimize_2.o -I./ $(cuda_lib) -lcudart
	nvcc --compile src/layer/conv_gpu_optimize_3.cc -o src/layer/conv_gpu_optimize_3.o -I./ $(cuda_lib) -lcudart

loss.o:
	nvcc --compile src/loss/cross_entropy_loss.cc -o src/loss/cross_entropy_loss.o -I./ $(cuda_lib) -lcudart
	nvcc --compile src/loss/mse_loss.cc -o src/loss/mse_loss.o -I./ $(cuda_lib) -lcudart

optimizer.o:
	nvcc --compile src/optimizer/sgd.cc -o src/optimizer/sgd.o -I./ $(cuda_lib) -lcudart

gpu.o:
	nvcc --compile src/gpu_src/gpu_utils.cu -o src/gpu_src/gpu_utils.o -I./ $(cuda_lib) -lcudart

#--------------------------------------------------------------------------------------------------------------------------------------------------

basic.o: 
	rm -f src/gpu_src/gpu_forward_basic.o 
	nvcc --compile src/gpu_src/gpu_forward_basic.cu -o src/gpu_src/gpu_forward_basic.o -I./ $(cuda_lib) -lcudart

optimize_0.o:
	rm -f src/gpu_src/gpu_forward_optimize_0.o 
	nvcc --compile src/gpu_src/gpu_forward_optimize_0.cu -o src/gpu_src/gpu_forward_optimize_0.o -I./ $(cuda_lib) -lcudart

optimize_1.o:
	rm -f src/gpu_src/gpu_forward_optimize_1.o 
	nvcc --compile src/gpu_src/gpu_forward_optimize_1.cu -o src/gpu_src/gpu_forward_optimize_1.o -I./ $(cuda_lib) -lcudart

optimize_2.o:
	rm -f src/gpu_src/gpu_forward_optimize_2.o
	nvcc --compile src/gpu_src/gpu_forward_optimize_2.cu -o src/gpu_src/gpu_forward_optimize_2.o -I./ $(cuda_lib) -lcudart

optimize_3.o:
	rm -f src/gpu_src/gpu_forward_optimize_3.o
	nvcc --compile src/gpu_src/gpu_forward_optimize_3.cu -o src/gpu_src/gpu_forward_optimize_3.o -I./ $(cuda_lib) -lcudart

#-----------------------------------------------------------------------------------------------------------------------------------------------------
host_convolution:
	nvcc --compile host.cc -I./ $(cuda_lib) -lcudart
	nvcc -o host -lm -lcuda -lrt host.o src/network.o src/fashion_mnist.o src/layer/*.o src/gpu_src/*.o -I./ $(cuda_lib) -lcudart
	./host

kernel_convolution_basic:
	nvcc --compile gpu_basic.cc -I./ $(cuda_lib) -lcudart
	nvcc -o gpu_basic -lm -lcuda -lrt gpu_basic.o src/network.o src/fashion_mnist.o src/layer/*.o src/gpu_src/*.o -I./ $(cuda_lib) -lcudart
	./gpu_basic

kernel_convolution_optimize_0:
	nvcc --compile gpu_optimize_0.cc -I./ $(cuda_lib) -lcudart
	nvcc -o gpu_optimize_0 -lm -lcuda -lrt gpu_optimize_0.o src/network.o src/fashion_mnist.o src/layer/*.o src/gpu_src/*.o -I./ $(cuda_lib) -lcudart
	./gpu_optimize_0

kernel_convolution_optimize_1:
	nvcc --compile gpu_optimize_1.cc -I./ $(cuda_lib) -lcudart
	nvcc -o gpu_optimize_1 -lm -lcuda -lrt gpu_optimize_1.o src/network.o src/fashion_mnist.o src/layer/*.o src/gpu_src/*.o -I./ $(cuda_lib) -lcudart
	./gpu_optimize_1

kernel_convolution_optimize_2:
	nvcc --compile gpu_optimize_2.cc -I./ $(cuda_lib) -lcudart
	nvcc -o gpu_optimize_2 -lm -lcuda -lrt gpu_optimize_2.o src/network.o src/fashion_mnist.o src/layer/*.o src/gpu_src/*.o -I./ $(cuda_lib) -lcudart
	./gpu_optimize_2

kernel_convolution_optimize_3:
	nvcc --compile gpu_optimize_3.cc -I./ $(cuda_lib) -lcudart
	nvcc -o gpu_optimize_3 -lm -lcuda -lrt gpu_optimize_3.o src/network.o src/fashion_mnist.o src/layer/*.o src/gpu_src/*.o -I./ $(cuda_lib) -lcudart
	./gpu_optimize_3

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

train_model: 
	nvcc --compile build_model.cc -I./ $(cuda_lib) -lcudart
	nvcc -o build_model -lm -lcuda -lrt build_model.o src/network.o src/fashion_mnist.o src/layer/*.o src/loss/*.o src/optimizer/*.o src/gpu_src/*.o -I./ $(cuda_lib) -lcudart
	./build_model

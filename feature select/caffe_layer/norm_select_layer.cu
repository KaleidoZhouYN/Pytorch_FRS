#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/norm_select_layer.hpp"

namespace caffe {
	
template <typename Dtype>
__global__ void kernel_channel_sum(const int num, const int channels,const int spatial_dim, Dtype epsilon,
								const Dtype* data, Dtype* norm_data) {
	CUDA_KERNEL_LOOP(index, num * spatial_dim) {
		int n = index / spatial_dim; 
		int s = index % spatial_dim; 
		Dtype sum = 0; 
		for (int c = 0; c < channels; ++c) {
			sum += data[(n * channels + c) * spatial_dim + s]; 
		}
		norm_data[index] = sum + epsilon; 
	}
}

template <typename Dtype>
__global__ void kernel_norm_select(const int num, const int channels,const int spatial_dim,
								const Dtype* norm_data,Dtype* data,const Dtype threshold) {
    CUDA_KERNEL_LOOP(index,num * spatial_dim) {
		int n = index / spatial_dim; 
		int s = index % spatial_dim; 
		data[n] = (norm_data[n*spatial_dim+s] < threshold)? Dtype(0) : Dtype(1); 
	}
}

template <typename Dtype>
__global__ void kernel_forward(const int num,const int channels,const int spatial_dim,
								const Dtype* norm_data,const Dtype* bottom_data,Dtype* top_data,const Dtype threshold){
	CUDA_KERNEL_LOOP(index,num*spatial_dim) {
		int n = index / spatial_dim; 
		int s = index % spatial_dim;
		if (norm_data[n*spatial_dim + s] < threshold)
		{
			for (int c = 0; c < channels; ++c) 
				top_data[(n*channels+c)*spatial_dim + s] = Dtype(0); 
		}
		else{
			for (int c = 0; c < channels; ++c) 
				top_data[(n*channels+c)*spatial_dim + s] = bottom_data[(n*channels+c)*spatial_dim + s]; 			
		}
	}
}

template <typename Dtype>
__global__ void kernel_backward(const int num,const int channels,const int spatial_dim,
						const Dtype* norm_data,const Dtype* top_diff,Dtype* bottom_diff,const Dtype threshold){
	CUDA_KERNEL_LOOP(index,num*spatial_dim){
		int n = index / spatial_dim; 
		int s = index % spatial_dim; 
		if (norm_data[n*spatial_dim + s] < threshold)
		{
			for (int c = 0; c < channels; ++c)
				bottom_diff[(n*channels+c)*spatial_dim + s] = Dtype(0); 
		}
		else 
		{
			for (int c = 0; c < channels; ++c)
				bottom_diff[(n*channels+c)*spatial_dim + s] = top_diff[(n*channels+c)*spatial_dim + s]; 
		}
	}
}

template <typename Dtype>
void NormSelectLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	
	const Dtype* bottom_data = bottom[0]->gpu_data(); 
	Dtype* top_data = top[0]->mutable_gpu_data(); 
	Dtype* square_data = squared_.mutable_gpu_data(); 
	Dtype* norm_data = norm_.mutable_gpu_data(); 
	
	int num = bottom[0]->num(); 
	int channels = bottom[0]->channels(); 
	int spatial_dim = bottom[0]->height() * bottom[0]->width(); 
	
	caffe_gpu_powx(num*channels*spatial_dim, bottom_data, Dtype(2),square_data); 
	kernel_channel_sum<Dtype> << <CAFFE_GET_BLOCKS(num*spatial_dim),
		CAFFE_CUDA_NUM_THREADS >> >(num, channels, spatial_dim, 1e-12, square_data, norm_data); 
	caffe_gpu_powx(num * spatial_dim, norm_data, Dtype(-0.5), norm_data);
	kernel_norm_select<Dtype> << <CAFFE_GET_BLOCKS(num*spatial_dim),
		CAFFE_CUDA_NUM_THREADS >> >(num, channels, spatial_dim, norm_data, top[1]->mutable_gpu_data(),threshold_); 
	kernel_forward<Dtype> << <CAFFE_GET_BLOCKS(num*spatial_dim),
		CAFFE_CUDA_NUM_THREADS >> >(num,channels,spatial_dim,norm_data,bottom_data,top_data,threshold_);
	
}

template <typename Dtype>
void NormSelectLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down,const vector<Blob<Dtype>*>& bottom) {
	
	const Dtype* top_diff = top[0]->gpu_diff(); 
	const Dtype* norm_data = norm_.gpu_data(); 
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff(); 
	
	int num = top[0]->num(); 
	int channels = top[0]->channels(); 
	int spatial_dim = bottom[0]->height() * bottom[0]->width(); 
	
	kernel_backward<Dtype> << <CAFFE_GET_BLOCKS(num*spatial_dim),
		CAFFE_CUDA_NUM_THREADS >> >(num,channels,spatial_dim,norm_data,top_diff,bottom_diff,threshold_); 
}

INSTANTIATE_LAYER_GPU_FUNCS(NormSelectLayer); 

} // namespace caffe

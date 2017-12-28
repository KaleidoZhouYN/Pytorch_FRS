#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/layer.hpp" 
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/norm_select_layer.hpp"

namespace caffe {
	
// if norm(x[i]) < threshold then x = 0

template <typename Dtype>
void NormSelectLayer<Dtype>::LayerSetUp(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		threshold_ = this->layer_param_.normselect_param().threshold();
}

template <typename Dtype>
void NormSelectLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
		
	top[0]->Reshape(bottom[0]->num(),bottom[0]->channels(),
		bottom[0]->height(),bottom[0]->width());
	if (top.size() == 2) {
		top[1]->Reshape(vector<int>(bottom[0]->num(),1));
	}
	norm_.Reshape(bottom[0]->num(),1,
					bottom[0]->height(),bottom[0]->width()); 
					
}

template <typename Dtype>
void NormSelectLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	
	const Dtype* bottom_data = bottom[0]->cpu_data(); 
	Dtype* top_data = top[0]->mutable_cpu_data(); 
	Dtype* square_data = squared_.mutable_cpu_data(); 
	Dtype* norm_data = norm_.mutable_cpu_data(); 
	int num = bottom[0]->num(); 
	int channels = bottom[0]->channels(); 
	int spatial_dim = bottom[0]->height() * bottom[0]->width(); 
	
	caffe_sqr<Dtype>(num*channels*spatial_dim,bottom_data,square_data);
	for (int n = 0; n < num; n++) {
		for (int s = 0; s < spatial_dim; s++) {
			norm_data[n*spatial_dim + s] = Dtype(0); 
			for (int c = 0; c < channels; c++) {
				norm_data[n*spatial_dim + s] += square_data[(n*channels+c) * spatial_dim + s]; 
			}
			norm_data[n*spatial_dim + s] += 1e-6; 
			norm_data[n*spatial_dim + s] = sqrt(norm_data[n*spatial_dim + s]);
			if (norm_data[n*spatial_dim + s] < threshold_)
			{
				for (int c = 0; c < channels; c++)
					top_data[(n*channels+c)*spatial_dim+s] = Dtype(0); 
				if (top.size() == 2)
					top[1]->mutable_cpu_data()[n] = Dtype(0);
			}
			else
			{
				for (int c = 0; c < channels; c++)
					top_data[(n*channels+c)*spatial_dim+s] = bottom_data[(n*channels+c)*spatial_dim+s];
				if (top.size() == 2)
					top[1]->mutable_cpu_data()[n] = Dtype(1);
			}
		}
	}
	
}

template <typename Dtype>
void NormSelectLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down,const vector<Blob<Dtype>*>& bottom) {
	
	const Dtype* top_diff = top[0]->cpu_diff(); 
	const Dtype* norm_data = norm_.cpu_data(); 
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff(); 
	
	int num = bottom[0]->num(); 
	int channels = bottom[0]->channels(); 
	int spatial_dim = bottom[0]->height() * bottom[0]->width(); 
	
	for (int n = 0; n < num; ++n) {
		for (int s = 0; s < spatial_dim; s++) {
			if (norm_data[n*spatial_dim+s] <threshold_)
			{
				for (int c = 0; c < channels; c++)
					bottom_diff[(n*channels+c)*spatial_dim+s] = Dtype(0); 
			}
			else 
			{
				for (int c = 0; c < channels; c++)
					bottom_diff[(n*channels+c)*spatial_dim+s] = top_diff[(n*channels+c)*spatial_dim+s]; 				
			}
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU(NormSelectLayer); 
#endif

INSTANTIATE_CLASS(NormSelectLayer); 
REGISTER_LAYER_CLASS(NormSelect); 

}  // namespace caffe
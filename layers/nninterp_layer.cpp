#include <vector>
#include <algorithm>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {
template<typename Dtype>
void NNInterpLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *>& bottom , const vector<Blob<Dtype> *>& top){
	NNInterpParameter nninterp_param = this->layer_param_.nninterp_param();
  int num_specs = 0;
  num_specs += nninterp_param.has_zoom_factor();
  num_specs += nninterp_param.has_shrink_factor();
  num_specs += nninterp_param.has_height() && nninterp_param.has_width();
  CHECK_EQ(num_specs, 1) << "Output dimension specified either by "
			 << "zoom factor or shrink factor or explicitly";
}


template<typename Dtype>
void NNInterpLayer<Dtype>::Reshape(const vector<Blob<Dtype> *>& bottom , const vector<Blob<Dtype> *>& top){
	int num = bottom[0]->num();
	int channel = bottom[0]->channels();
	int height_in = bottom[0]->height();
	int width_in = bottom[0]->width();

	NNInterpParameter nninterp_param = this->layer_param_.nninterp_param();
	if(nninterp_param.has_zoom_factor()){
		int zoom_factor = nninterp_param.zoom_factor();
		CHECK_GE(zoom_factor  , 1)<<"Zoom factor must be positive!";
		height_out_ = height_in + (height_in-1)*(zoom_factor - 1);
		width_out_ = width_in + (width_in-1)*(zoom_factor - 1);
	}
	else if(nninterp_param.has_shrink_factor()){
		int shrink_factor = nninterp_param.shrink_factor();
		CHECK_GE(shrink_factor  , 1)<<"Shrink factor must be positive!";
		height_out_ = (height_in - 1) / shrink_factor + 1;
		width_out_ = (width_in - 1) / shrink_factor + 1;
	}
	else if(nninterp_param.has_height() && nninterp_param.has_width()){
		height_out_ = nninterp_param.height();
		width_out_ = nninterp_param.width();
	}
	else{
		LOG(FATAL);
	}
	top[0]->Reshape(num , channel , height_out_ , width_out_);
}

template<typename Dtype>
void NNInterpLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *>& bottom ,const vector<Blob<Dtype> *>& top){
	
	int num = bottom[0]->num();
	int channel = bottom[0]->channels();
	int height_in = bottom[0]->height();
	int width_in = bottom[0]->width();
	
	float scale_h = static_cast<float>(height_in) / height_out_;
	float scale_w = static_cast<float>(width_in) / width_out_;
	
	int dst_feature_size_2d = height_out_ * width_out_;
	int dst_feature_size_3d = channel * dst_feature_size_2d;
	
	int src_feature_size_2d = height_in *width_in;
	int src_feature_size_3d = channel * src_feature_size_2d;

	for(int n = 0 ; n < num ; n++){
		for(int c = 0 ; c < channel ; c++){
			for(int h = 0 ; h < height_out_ ; h++){
				int src_h = std::min((int)(h * scale_h) , height_in-1);
				for(int w = 0 ; w < width_out_ ;w++){
					int src_w = std::min((int)(w * scale_w) , width_in - 1);
					top[0]->mutable_cpu_data()[n*dst_feature_size_3d + c*dst_feature_size_2d + h*width_out_ + w] = 
						bottom[0]->mutable_cpu_data()[n*src_feature_size_3d + c*src_feature_size_2d + src_h * width_in + src_w];
				}
			}
		}
	}
	return;
}

template<typename Dtype>
void NNInterpLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *>& top ,	const vector<bool> &propagate_down , const vector<Blob<Dtype> *>& bottom){
	if(!propagate_down[0]){
		return ;
	}
	caffe_set(bottom[0]->count() , Dtype(0) , bottom[0]->mutable_cpu_diff());

	int num = bottom[0]->num();
	int channel = bottom[0]->channels();
	int height_in = bottom[0]->height();
	int width_in = bottom[0]->width();
	
	float scale_h = static_cast<float>(height_in) / height_out_;
	float scale_w = static_cast<float>(width_in) / width_out_;
	
	int dst_feature_size_2d = height_out_ * width_out_;
	int dst_feature_size_3d = channel * dst_feature_size_2d;
	
	int src_feature_size_2d = height_in *width_in;
	int src_feature_size_3d = channel * src_feature_size_2d;

	for(int n = 0 ; n < num ; n++){
		for(int c = 0 ; c < channel ; c++){
			for(int h = 0 ; h < height_out_ ; h++){
				int src_h = std::min((int)(h*scale_h) , height_in-1);
				for(int w = 0 ; w < width_out_ ;w++){
					int src_w = std::min((int)(w * scale_w) , width_in - 1);
					bottom[0]->mutable_cpu_diff()[n*src_feature_size_3d + c*src_feature_size_2d + src_h*width_in + src_w] += 
						top[0]->cpu_diff()[n*dst_feature_size_3d + c*dst_feature_size_2d + h * height_out_ + w];
				}
			}
		}
	}
	return;
}
#ifdef CPU_ONLY
STUB_GPU(NNInterpLayer);
#endif

INSTANTIATE_CLASS(NNInterpLayer);
REGISTER_LAYER_CLASS(NNInterp);

}

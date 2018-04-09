#include <vector>

#include "caffe/util/gpu_util.cuh"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"


namespace caffe {

template <typename Dtype>
__global__ void NNInterpForward(const int nthreads,
    const Dtype* const bottom_data, const int num ,  const int channels, const int height_in, const int width_in, 
		const float scale_h , const float scale_w ,
		Dtype* const top_data , const int height_out , const int width_out) {

  CUDA_KERNEL_LOOP(index, nthreads) {
		
		int n = index / width_out / height_out / channels ;

		int c = (index / width_out / height_out ) % channels ;

		int h = (index / width_out ) % height_out ;
		int w = index % width_out ;
		
		int src_h =(int)(h*scale_h);
		src_h = (src_h > (height_in - 1 ) ?( height_in - 1) : src_h) ;
		int src_w = (int)(w*scale_w);
		src_w = (src_w > (width_in - 1) ? width_in - 1 : src_w );

		int src_index = n*height_in*width_in*channels + c*height_in*width_in + src_h*width_in + src_w;

		top_data[index] = bottom_data[src_index];
	}
}

template <typename Dtype>
void NNInterpLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	
	int count = top[0]->count();
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int height_in = bottom[0]->height();
	int width_in = bottom[0]->width();
	
	int height_out = top[0]->height();
	int width_out = top[0]->width();

	float scale_h = (float) height_in / height_out ;
	float scale_w = (float) width_in / width_out ;

	NNInterpForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
				count , bottom_data ,num , channels , height_in , width_in ,
				scale_h , scale_w , 
				top_data , height_out , width_out	);

  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void NNInterpBackward(const int nthreads,
    Dtype* const bottom_diff, const int num ,  const int channels, const int height_in, const int width_in, 
		const float scale_h , const float scale_w ,
		const Dtype* const top_diff , const int height_out , const int width_out) {

  CUDA_KERNEL_LOOP(index, nthreads) {
		int n = index / width_out / height_out / channels ;

		int c = (index / width_out / height_out ) % channels ;

		int h = (index / width_out ) % height_out ;
		int w = index % width_out ;
		
		int src_h =(int)(h*scale_h);
		src_h = (src_h > (height_in - 1 ) ?( height_in - 1) : src_h) ;
		int src_w = (int)(w*scale_w);
		src_w = (src_w > (width_in - 1) ? width_in - 1 : src_w );

		int src_index = n*height_in*width_in*channels + c*height_in*width_in + src_h*width_in + src_w;
		// bottom_diff[src_index] += top_diff[index]; 
        caffe_gpu_atomic_add(top_diff[index], bottom_diff + src_index);
	}
}

template <typename Dtype>
void NNInterpLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, 
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if(!propagate_down[0]){
		return ;
	}
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	const Dtype* top_diff = top[0]->gpu_diff();
	caffe_gpu_set(bottom[0]->count(), Dtype(0) , bottom_diff);

	int count = top[0]->count();
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int height_in = bottom[0]->height();
	int width_in = bottom[0]->width();
	
	int height_out = top[0]->height();
	int width_out = top[0]->width();

	float scale_h = (float) height_in / height_out ;
	float scale_w = (float) width_in / width_out ;

	NNInterpBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
				count , bottom_diff ,num , channels , height_in , width_in ,
				scale_h , scale_w , 
				top_diff , height_out , width_out	);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(NNInterpLayer);

}  // namespace caffe

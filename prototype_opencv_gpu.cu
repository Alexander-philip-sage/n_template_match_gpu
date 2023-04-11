//https://opencv.org/platforms/cuda/
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"

struct coordinates {
    int sec1_x;
    int sec1_y;
    int sec2_x;
    int sec2_y;
}
struct template_match_respons {
    //all the params that are passed in and modified 
    //  as the result of the template match
    // Initialize the results matrix in GPU land
}

__global__ void thread_template_match(const? SOMETHING coords_gpu, cv::gpu::GpuMat sec1_gpu, cv::gpu::GpuMat sec2_gpu, SOMETHING resp_gpu) {
//https://github.com/NVIDIA/cuda-samples/blob/master/Samples/0_Introduction/vectorAdd/vectorAdd.cu
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < numElements) {
    templ = crop(sec1_gpu, coords_gpu[i].sec1_x, coords_gpu[i].sec1_y)
    window = crop(sec2_gpu, coords_gpu[i].sec2_x, coords_gpu[i].sec2_y)
    //static bool matchTemplate_CCOEFF(InputArray _image, InputArray _templ, OutputArray _result)
    resp_gpu[i].res = template_match(window, templ, resp_gpu[i].proximity_map)
  }
}

int main (int argc, char* argv[])
{
    try
    {
        //take in the coords and filepaths through cython interface
        cv::SOMETHING coords_cpu
        cv::SOMETHING resp_cpu
        cv::Mat sec1_cpu = cv::imread(sec1_path, CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat sec2_cpu = cv::imread(sec2_path, CV_LOAD_IMAGE_GRAYSCALE);
        //https://docs.opencv.org/3.4/d0/d60/classcv_1_1cuda_1_1GpuMat.html#a45500c2bf2a548cb8c27231d5f9d697b
        //if initialize with data, then we don't need to use the non-blocking upload call
        cv::gpu::GpuMat sec1_gpu(sec1_cpu);
        cv::gpu::GpuMat sec2_gpu(sec2_cpu);
        cv::gpu::SOMETHING coords_gpu = 
        int result_cols =  sec2_cpu.cols - sec1_cpu.cols + 1;
        int result_rows = sec2_cpu.rows - sec1_cpu.rows + 1;
        cv::Mat result_cpu( result_rows, result_cols, CV_32FC1 );
        cv::cuda::GpuMat result_gpu( result_rows, result_cols, CV_32FC1 );
        //sec1_gpu.upload(sec1_cpu);
        //sec2_gpu.upload(sec2_cpu);
        coords_gpu.upload(coords_cpu);

        int threadsPerBlock = 256;
        int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
        printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
                threadsPerBlock);
        thread_template_match<<<blocksPerGrid, threadsPerBlock>>>(coords_gpu, sec1_gpu, sec2_gpu, resp_gpu);
        //someone tried this
        //https://towardsdatascience.com/opencv-gpu-usage-disappoints-bc331329932d
        cv::cuda::TemplateMatching tm(cv::CV_TM_CCOEFF_NORMED);
        tm->match(img_gpu, template_gpu, result_gpu );
        //opencv example
        //cv::gpu::threshold(src, dst, 128.0, 255.0, CV_THRESH_BINARY);

        resp_gpu.download(resp_cpu);
        //pass back respons through cython interface
        ~resp_gpu; ~coords_gpu; ~sec1_gpu; ~sec2_gpu;
    }
    catch(const cv::Exception& ex)
    {
        std::cout << "Error: " << ex.what() << std::endl;
    }
    return 0;
}
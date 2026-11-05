#ifndef _Decoder1Net_H_
#define _Decoder1Net_H_

#include <string>
#include <vector>

#include <cudnn.h>

#include "helper.h"
#include "network.h"

namespace CUDA_NAMESPACE {



class Decoder1Net
{
    public:
    Decoder1Net();


    
    CuTensor<float> *forward();
    int load_pretrain();
    int write_file();


    void cuda();
    void test();

    int set_input_shape(const std::vector<int> &input_shape);

    std::unique_ptr<CuTensor<float>> output_;
    std::unique_ptr<CuTensor<float>> input_;
    CuTensor<float> *output_ptr_;
    CuTensor<float> *input_ptr_;

    std::vector<int> input_shape_;
    std::vector<int> output_shape_;


  private:
    Network model;

};

} // namespace cudl


#endif // _Decoder1Net_H_
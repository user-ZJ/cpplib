#ifndef _NETWORK_H_
#define _NETWORK_H_

#include <string>
#include <vector>

#include <cudnn.h>

#include "helper.h"
#include "layer.h"

namespace CUDA_NAMESPACE {

typedef enum {
    training,
    inference
} WorkloadType;

class Network
{
    public:
    Network();
    ~Network();

    void add_layer(Layer *layer);
    CuTensor<float> *forward(CuTensor<float> *input);
    int load_pretrain();
    int write_file();


    void cuda();
    void test();

    int set_input_shape(const std::vector<int> &input_shape);

    std::unique_ptr<CuTensor<float>> output_;
    std::unique_ptr<CuTensor<float>> input_;
    CuTensor<float> *output_ptr_;
    CuTensor<float> *input_ptr_;

    std::vector<Layer *> layers();

    std::vector<int> input_shape_;
    std::vector<int> output_shape_;


  private:
    std::vector<Layer *> layers_;

    CudaContext *cuda_ = nullptr;

    WorkloadType phase_ = inference;
};

} // namespace cudl


#endif // _NETWORK_H_
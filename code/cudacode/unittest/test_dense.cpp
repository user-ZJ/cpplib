#include "layers/Dense.h"
#include <vector>
#include <iostream>

using namespace CUDA_NAMESPACE;

int main(int argc,char *argv[]){
    CudaContext context;
    Dense dense("dense1",3,2);
    std::vector<float> input{1,1,1,2,2,2,3,3,3};
    std::vector<float> params{1,1,2,2,3,3,3,2,1};
    std::vector<char> buff(params.size()*sizeof(float));
    memcpy(buff.data(),params.data(),buff.size());
    dense.load_parameter(buff);
    NDTensor input_t({3,3}),output_t({2,3});
    memcpy(input_t.data<float>(),input.data(),input_t.byteSize());
    input_t.dump2File<float>("input.txt");
    NDTensor input_t1 = input_t.cuda();
    NDTensor output_t1 = output_t.cuda();
    int device_type = static_cast<int>(input_t1.getDeviceType());
    std::cout<<"input_t1:"<<input_t1.data<float>()<<std::endl;
    std::cout<<"output_t1:"<<output_t1.data<float>()<<std::endl;
    std::cout<<"device_type:"<<device_type<<std::endl;
    dense.set_input_shape(std::vector<int>{3,3});
    dense.forward(context,&input_t1,&output_t1);
    output_t1.cpu().dump2File<float>("output.txt");


    return 0;

}
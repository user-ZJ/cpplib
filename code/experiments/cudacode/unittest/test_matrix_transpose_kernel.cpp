#include <vector>
#include <iostream>
#include "base/NDTensor.h"
#include "kernels/matrix_transpose.h"

using namespace CUDA_NAMESPACE;

int main(int argc,char *argv[]){
    int M = 36,N = 42;
    NDTensor t({M,N}),out({N,M});
    std::vector<float> d(t.size());
    for(int i=0;i<M*N;i++)
        d[i]=i;
    memcpy(t.data<float>(),d.data(),t.byteSize());
    t.dump2File<float>("input.txt");
    t = t.cuda();
    out = out.cuda();
    matrix_transpose(t.data<float>(),out.data<float>(),M,N);
    out.cpu().dump2File<float>("output.txt");
    return 0;

}
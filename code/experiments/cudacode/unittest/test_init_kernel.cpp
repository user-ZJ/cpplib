#include <vector>
#include <iostream>
#include "base/NDTensor.h"
#include "kernels/init_kernel.h"

using namespace CUDA_NAMESPACE;

int main(int argc,char *argv[]){
    int N = 67;
    NDTensor t({1,N});
    t = t.cuda();
    init_vec(t.data<float>(),N,1.5);
    t.cpu().dump2File<float>("init.txt");
    return 0;

}
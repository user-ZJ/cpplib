#include <iostream>
#include <vector>
#include "utils/NDTensor.h"

using namespace BASE_NAMESPACE;

int main() {
    int arr[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    NDTensor t({1,2,4,3});
    std::cout<<t.at<float>({0,0,0,0})<<std::endl;
    std::cout<<t.at<int>({0,0,0,0})<<std::endl;
    std::cout<<t.data<float>()<<std::endl;
    std::cout<<t.data<int>()<<std::endl;
    return 0;
}
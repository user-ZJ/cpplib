#include <iostream>
#include <vector>

template<typename T>
T& at(const std::initializer_list<int>& indexs) {
    static T a=0;
    return a;
}

int main() {
    int arr[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int& result1 = at<int>({0}); // 调用模板函数，返回arr[0]
    float& result1 = at<float>({0}); // 调用模板函数，返回arr[0]
    return 0;
}
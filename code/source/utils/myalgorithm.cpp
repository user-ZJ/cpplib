#include "myalgorithm.h"
#include <iterator>

namespace BASE_NAMESPACE{

template<typename Container,typename F>
Container topk(Container &c,int k,F compare){
    
}



template <typename Iter>
void quickSort(Iter begin, Iter end,int distance)
{
    if (distance <= 1) {
        return;
    }
    // 选择基准值位置
    auto pivotValue = *begin;
    // 将元素分为小于和大于基准值的两个子列表
    int left_distance=0,right_distance=0;
    auto left = begin;
    auto right = std::prev(end);
    while (left != right) {
        while (*right >= pivotValue && right != left) {
            --right;
            ++right_distance;
        }
        while (*left <= pivotValue && left != right) {
            ++left;
            ++left_distance;
        }
        
        if (left != right) {
            std::swap(*left, *right);
        }
    }
    std::swap(*begin, *right);

    // 递归调用快速排序，对小于和大于基准值的部分进行排序
    quickSort<Iter>(begin, left,left_distance);
    quickSort<Iter>(++right, end,right_distance);
}

}
#include <list>

namespace BASE_NAMESPACE{

// 使用快排思想算法获取容器中topk个元素，topk中元素不是排序的
// 容器需要支持迭代器的++和--操作
template<typename Container,typename F>
Container topk(Container &c,int k,F compare);



// 快速排序算法
template <typename Iter>
void quickSort(Iter begin, Iter end,int distance);

template<typename Container>
void quickSort(Container &c){
    quickSort(c.begin(),c.end(),c.size());
}
template void quickSort<std::list<int>>(std::list<int> &c);

}
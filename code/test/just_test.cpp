#include <cassert>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

template <typename Comparable>
class BinaryHeap {
 public:
  explicit BinaryHeap(int capacity = 100);
  explicit BinaryHeap(const std::vector<Comparable> &items);
  bool isEmpty() const {
    return currentSize == 0;
  }
  const Comparable &findMin() const {
    return array[currentSize];
  }
  void insert(const Comparable &x);
  void deleteMin();
  // 最小值保存到minItem，并删除最小值
  void deleteMin(Comparable &minItem);
  void makeEmpty() {
    currentSize = 0;
  }

 private:
  int currentSize;
  std::vector<Comparable> array;
  void buildHeap();
  void percolateDown(int hole);
};

template <typename Comparable>
BinaryHeap<Comparable>::BinaryHeap(int capacity) {
  currentSize = 0;
  array.resize(capacity);
}

template <typename Comparable>
BinaryHeap<Comparable>::BinaryHeap(const std::vector<Comparable> &items) {
  currentSize = items.size();
  array.resize(items.size()+100);
  for(auto &cmp:items){
    array[++currentSize] = cmp;
  }
  buildHeap();
}

template <typename Comparable>
void BinaryHeap<Comparable>::insert(const Comparable &x) {
  if (currentSize == array.size() - 1) array.resize(array.size() * 2);
  // percolate up
  int hole = ++currentSize;
  for (; hole > 1 && x < array[hole / 2]; hole /= 2) {
    array[hole] = array[hole / 2];
  }
  array[hole] = x;
}

template <typename Comparable>
void BinaryHeap<Comparable>::deleteMin() {
  if (isEmpty()) return;
  array[1] = array(currentSize--);
  percolateDown(1);
}

template <typename Comparable>
void BinaryHeap<Comparable>::deleteMin(Comparable &minItem) {
  if (isEmpty()) return;
  minItem = array[1];
  array[1] = array(currentSize--);
  percolateDown(1);
}

template <typename Comparable>
void BinaryHeap<Comparable>::percolateDown(int hole) {
  int child;
  Comparable tmp = array[hole];
  for (; hole * 2 < currentSize; hole = child) {
    child = hole * 2;
    if (child != currentSize && array[child] > array[child + 1]) { child++; }
    if (array[child] < tmp) {
      array[hole] = array[child];
    } else {
      break;
    }
  }
  array[hole] = tmp;
}

template <typename Comparable>
void BinaryHeap<Comparable>::buildHeap(){
  for(int i=currentSize/2;i>=1;i--)
    percolateDown(i);
}

class Test{};

int main(int argc, char *argv[]) {
  int i;
  const std::type_info &info1 = typeid(i);
  std::cout<<info1.name()<<std::endl;

  double j;
  const std::type_info &info2 = typeid(j);
  std::cout<<info2.name()<<std::endl;

  Test t;
  const std::type_info &info3 = typeid(t);
  std::cout<<info3.name()<<std::endl;

  Test *t1;
  const std::type_info &info4 = typeid(t1);
  std::cout<<info4.name()<<std::endl;
}

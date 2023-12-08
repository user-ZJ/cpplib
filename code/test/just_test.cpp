#include <iostream>
#include <list>
#include "utils/myalgorithm.h"

using namespace BASE_NAMESPACE;

// 输出函数
template <typename T>
void printList(const std::list<T>& lst)
{
    for (const auto& element : lst) {
        std::cout << element << " ";
    }
    std::cout << std::endl;
}

int main()
{
    std::list<int> lst = {5, 1, 9, 3, 7, 4};
    std::cout << "Before sorting: ";
    printList(lst);

    quickSort(lst);

    std::cout << "After sorting: ";
    printList(lst);

    return 0;
}
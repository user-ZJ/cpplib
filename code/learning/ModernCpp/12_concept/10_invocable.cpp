#include <concepts>
#include <functional>
#include <iostream>
#include <vector>

// template< class F, class... Args >
// concept invocable =
//   requires(F&& f, Args&&... args) {
//     std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
//   };




// template <std::invocable<int> F>
// void PrintVec(const std::vector<int>& vec, F fn) {
//     for (auto &elem : vec)
//         std::cout << fn(elem) << '\n';
// }


void PrintVec(const std::vector<int>& vec, std::invocable<int> auto fn) {
    for (auto &elem : vec)
        std::cout << fn(elem) << '\n';
}


int main() {
    std::vector ints { 1, 2, 3, 4, 5};
    PrintVec(ints, [](int v) { return -v; });
}
#include<iostream>

using namespace std;

template<typename T>
concept CanCompute=requires(T x, T y) {
    x + y; // 支持+
    x - y; // 支持 -
    x * y;   //支持 * 
};


template<typename Coll>
concept MyConcept= requires {
    typename Coll::value_type::first_type; // 元素值要有first_type
    typename Coll::value_type::second_type; // 元素值要有second_type
};


template<typename Coll>
concept MyConcept=requires(Coll::value_type v) {
    std::cout << v; // 元素支持<< 
};

template<typename T1, typename T2>
concept MyConcept=requires(T1 val, T2 p) {
    *p; // 支持*
    p[0]; // 支持索引
    p->value(); // 支持成员函数 value() without arguments has to be possible
    *p > val; // 支持*后和T1比较
    p == nullptr; // 支持和 nullptr比较
};


template<typename T1, typename T2>
concept MyConcept=requires(T1 val, T2 p) {
    *p > val; 
} || requires(T2 p) {  //取或
    p == nullptr; 
};



// template<typename T>
// concept MyConcept=requires {
//     std::integral<T>; //注意：并非要求 “是integral” 错误表达
//     ...
// };

template<typename T>
concept MyConcept=requires {
    requires std::integral<T>; //要求“是integral”
    ...
};

template<typename T>
concept MyConcept= std::integral<T> && //要求“是integral”
requires {
    ...
};



//类型需求
template<typename T1, typename T2>
concept MyConcept=requires {
    typename T1::value_type; 
    typename std::ranges::iterator_t<T1>; 
    typename std::ranges::iterator_t<std::vector<T1>>;
    typename std::common_type_t<T1, T2>; // T1 和 T2 有相同类型 
};

template<typename T>
concept MyConcept=requires(T x) {
    { &x } -> std::input_or_output_iterator;
    { x == x }
    { x == x } -> std::convertible_to<bool>;
    { x == x } noexcept
    { x == x } noexcept -> std::convertible_to<bool>;
};



//========
template<std::integral T> // 模板参数约束
class MyClass {
    ...
};
auto myFunc(const std::integral auto& val) {  //auto参数约束
    ...
};

template<typename T>
concept MyConcept = requires(T x) {
    { x + x } -> std::integral; // 复合定义
};

//约束返回值
std::integral auto foo() {

}


//约束可调用构造（函数指针、函数对象、lambda）
template<std::invocable<int, std::string> Callable>
void call(Callable op);

void call(std::invocable<int, std::string> auto op);


//auto使用的地方都可以用概念约束
std::integral auto val1 = 42; // OK
std::integral auto val2 = true; // 错误
for (const std::integral auto& elem : coll) {
    
}

//约束非类型参数
template<typename T, std::integral auto Max>
class SizedColl {
    ...
};




template<typename T>
concept GeoObject = requires(T obj) {
    { obj.width() } -> std::integral;
    { obj.height() } -> std::integral;
    obj.draw();
};









template<typename T>
concept ColoredGeoObject =
    GeoObject<T> && // 归并概念 GeoObject
    requires(T obj) { 
        obj.setColor(Color{});
        { obj.getColor() } -> std::convertible_to<Color>;
    };

template<GeoObject T>
void process(T) // 没有 setColor() 和 getColor() 优先
{
    ...
}

template<ColoredGeoObject T>
void process(T) // 必须有 setColor() 和 getColor()
{
    ...
}


template<typename I>
concept IsIterator =
    default_initializable<T> &&
    std::copyable<I> &&
    requires(I i) {
        typename iter_difference_t<I>;
        *i;
        { ++i } -> same_as<I&>;
        { i++ } -> same_as<I>;
    };
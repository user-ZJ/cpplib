# 1. vector

*vector*是一个能够存放任意类型的动态数组，能够增加和压缩数据，增加的时候容量不够，则以2倍的存储增加容量

https://blog.csdn.net/phoebin/article/details/3864590

## 1.1 创建

```cpp
#include <vector>
using std::vector;
vector<int> vInts;  //创建空vector
vector<int> vInts(100);  //创建容量为100的vector
vector<int> vInts(100，1); //创建容量为100的vector，并全部初始化为1
vector<int> vInts(vInts1);  //拷贝一个vector内容来创建一个vector
vector<int> vInts{1,2,3,4,5}; //  
//子序列
vector<int> sub_vec {int_vec.begin(), int_vec.begin()+5};
//创建矩阵
vector<vector<float>> result(10,vector<float>(20,0.0f));
```

## 1.2 增加/插入数据

```  cpp
#include <vector>
using std::vector;
vector<int> vInts;
vInts.push_back(9);  //在尾部增加数据
vInts.insert(vInts.begin()+1,6); //插入数据
```

## 1.3 获取容器大小

```cpp
#include <vector>
using std::vector;
vector<int> vInts;
vInts.empty()；   //判断是否为空
vInts.size()；   //返回容器中实际数据的个数。
vInts.capacity();    
```

## 1.4 访问数据

```cpp
#include <vector>
using std::vector;
vector<int> vInts(10,9);
vInts.at(2);   //推荐使用，at()进行了边界检查，如果访问超过了vector的范围，将抛出一个异常
vInts[2];    //不推荐使用，主要是为了与C语言进行兼容。它可以像C语言数组一样操作
```

## 1.5 删除数据

```cpp
#include <vector>
using std::vector;
vector<int> vInts(10,9);
vInts.erase(3);  //删除pos位置的数据
vInts.erase(vInts.begin(),vInts.end());  //删除pos位置的数据
vInts.pop_back();  //删除最后一个数据。
vInts.clear()();  //删除所有数据。
```

## 1.6 遍历

```cpp
#include <vector>
using std::vector;
vector<int> vInts(10,9);
for(vector<int>::iterator iter = vInts.begin(); iter != vInts.end(); iter++){
    cout<<*iter<<endl;
}
// c++ 11
for (auto i : vInts)
{
	cout << i<< endl;
}
```

## 1.7 查找

```cpp
#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;
int main(){
    vector<int> vInts(10,9);
    vInts.insert(vInts.begin()+3,6);
    vector<int>::iterator res = find(vInts.begin(),vInts.end(),6);                           
    if(res == vInts.end()){
        cout<<"not find\n";
    }else{
        cout<<"find "<<*res<<endl;
    }   
}
```

## 1.8 排序

```cpp
#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;
int main(){
    vector<int> vInts{1,3,2,5,4};
    vInts.insert(vInts.begin()+3,6);
    sort(vInts.begin(),vInts.end());  //从小到大
    sort(vInts.rbegin(),vInts.rend());  //从大到小
}
```

## 1.9 拼接

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
void show(vector<int> const &input) {
   for (auto const& i: input) {
      std::cout << i << " ";
   }
}
int main() {
   vector<int> v1 = { 1, 2, 3 };
   vector<int> v2 = { 4, 5 };
   v2.insert(v2.begin(), v1.begin(), v1.end());
   cout<<"Resultant vector is:"<<endl;
   show(v2);
   return 0;
}
```

```text
Resultant vector is:
1 2 3 4 5
```



# 2. List

list容器就是一个双向链表,可以高效地进行插入删除元素

注意：list的iterator是双向的，只支持++、--。如果要移动多个元素应该用next：

https://www.cnblogs.com/scandy-yuan/archive/2013/01/08/2851324.html

## 2.1 创建

```cpp
#include<iostream>
#include<list>
using namespace std;
int main(){
    list<int> c0; //空链表
    list<int> c1(3);  //建一个含三个默认值是0的元素的链表
    list<int> c2(5,2);  //建一个含五个元素的链表，值都是2
    list<int> c4(c2); //建一个c2的copy链表
    list<int> c5(c1.begin(),c1.end()); //c5含c1一个区域的元素[_First, _Last)  
    list<int> a1 {1,2,3,4,5};                                                             
    return 0;
}
```

## 2.2 增加/插入数据

```cpp
#include<iostream>
#include<list>
using namespace std;
int main(){
    list<int> a{1,2,3,4,5},a1;
    a1 = a;
    a1.assign(5,10);  //assign(n,num)      将n个num拷贝赋值给链表c。
    list<int>::iterator it;
    for(it = a1.begin();it!=a1.end();it++){
        cout << *it << "\t";
    }
    cout<<endl;
    a1.assign(a.begin(),a.end());   //assign(beg,end)      将[beg,end)区间的元素拷贝赋值给链表c。
    for(it = a1.begin();it!=a1.end();it++){
        cout << *it << "\t";
    }
    cout<<endl;
    a1.insert(a1.begin(),0);  //insert(pos,num)      在pos位置插入元素num。
    a1.insert(a1.begin(),2,88);  //insert(pos,n,num)      在pos位置插入n个元素num。
    int arr[5] = {11,22,33,44,55};
    a1.insert(a1.begin(),arr,arr+3);  //insert(pos,beg,end)      在pos位置插入区间为[beg,end)的元素。
    a1.push_front(9);  //push_front(num)      在开始位置增加一个元素。
    a1.push_back(99);  //push_back(num)      在末尾增加一个元素。
    return 0;
}
```

## 2.3 获取/修改容器大小

```cpp
//c.empty(); // 判断链表是否为空。
//c.size();  //返回链表c中实际元素的个数。
//c.max_size(); //返回链表c可能容纳的最大元素数量。
//resize(n)      从新定义链表的长度,超出原始长度部分用0代替,小于原始部分删除。
//resize(n,num)            从新定义链表的长度,超出原始长度部分用num代替。
#include<iostream>
#include<list>
using namespace std;
int main(){
    list<int> a{1,2,3,4,5},a1;
    cout<<a.empty()<<";"<<a.size()<<";"<<a.max_size()<<endl;
    return 0;
}
```



## 2.4 访问元素

```cpp
// c.front()      返回链表c的第一个元素。
// c.back()      返回链表c的最后一个元素。
#include <iterator>
#include<list>
using namespace std;
int main(){
    list<int> a1{1,2,3,4,5};
    list<int>::iterator it;
    it = next(a1.begin(),3);
    cout<<*it<<endl;
    a1.clear();
    return 0;
}
```

## 2.5 删除数据

```cpp
//c.clear();      清除链表c中的所有元素。
//c.erase(pos)　　　　删除pos位置的元素。
//c.pop_back()      删除末尾的元素。
//c.pop_front()      删除第一个元素。
//remove(num)             删除链表中匹配num的元素。
#include<iostream>
#include<list>
#include <iterator>
using namespace std;
int main(){
    list<int> a1{1,2,3,4,5};
    list<int>::iterator it;
    a1.erase(next(a1.begin(),3));
    a1.pop_front();
    a1.pop_back();
    
    for(it = a1.begin();it!=a1.end();it++){
        cout << *it << "\t";
    }
    cout<<endl;
    a1.clear();
    return 0;
}
```

## 2.6 遍历

```cpp
#include<iostream>
#include<list>
using namespace std;
int main(){
    list<int> a1 {1,2,3,4,5};
    //正向遍历
    list<int>::iterator it;
    for(it = a1.begin();it!=a1.end();it++){
        cout << *it << "\t";
    }
    cout<<endl;
    //反向遍历
    list<int>::reverse_iterator itr;
    for(itr = a1.rbegin();itr!=a1.rend();itr++){
        cout << *itr << "\t";
    }
    cout<<endl;
    return 0;
}

```

## 2.7 查找

```cpp
#include<iostream>
#include<list>
#include<algorithm>
using namespace std;
int main(){
    list<int> a1 {1,2,3,4,5};
    list<int>::iterator res = find(a1.begin(),a1.end(),3);                           
    if(res == a1.end()){
        cout<<"not find\n";
    }else{
        cout<<"find "<<*res<<endl;
    }   
}
```

## 2.8 翻转

```cpp
//reverse()       反转链表
list<int> a1{1,2,3,4,5};
a1.reverse();
```

## 2.9 排序

```cpp
// c.sort()       将链表排序，默认升序
// c.sort(comp)       自定义回调函数实现自定义排序
#include<iostream>
#include<list>
#include <iterator>
using namespace std;
int main(){
    list<int> a1{1,3,2,5,4};
    a1.sort();
    a1.sort([](int n1,int n2){return n1>n2;});
    list<int>::iterator it;
    for(it = a1.begin();it!=a1.end();it++){
        cout << *it << "\t";
    }
    cout<<endl;
    return 0;
}
```

## 2.10 去重

```cpp
#include<iostream>
#include<list>
#include <iterator>
using namespace std;
int main(){
    list<int> a1{1,1,2,2,3,4,5};
    a1.unique();     //去重
    list<int>::iterator it;
    for(it = a1.begin();it!=a1.end();it++){
        cout << *it << "\t";
    }
    cout<<endl;
    return 0;
}
```

# 3. map

## 3.1 创建

```cpp
#include <map>
map<int, string> mm;
//初始化列表来指定 map 的初始值
std::map<std::string, size_t> people{{"Ann", 25}, {"Bill", 46},{"Jack", 32},{"Jill", 32}};
std::map<std::string,size_t> people{std::make_pair("Ann",25),std::make_pair("Bill", 46),std::make_pair("Jack", 32),std::make_pair("Jill", 32)};
//移动和复制构造函数
std::map<std::string, size_t> personnel {people};
//用另一个容器的一段元素来创建一个 map
std::map<std::string, size_t> personnel {std::begin(people),std::end(people)};
```

## 3.2 增加/插入数据

```cpp
//第一种：用insert函数插入pair数据 ,如果key存在，插入失败
//第二种：用insert函数插入value_type数据，如果key存在，插入失败
//第三种：用数组方式插入数据，如果key存在，覆盖value
#include<iostream>
#include<map>
using namespace std;
int main(){
    map<int, string> mm;
    pair<map<int, string>::iterator, bool> Insert_Pair;
    Insert_Pair = mm.insert(pair<int,string>(0,"zero"));  //插入pair数据
    if(Insert_Pair.second == true)
        cout<<"Insert Successfully"<<endl;
    else
        cout<<"Insert Failure"<<endl;
    mm.insert(make_pair(1,"one"));        //插入pair数据
    mm.insert(map<int,string>::value_type(3,"three"));  //插入value_type数据
    mm[4] = "four";                  //数组方式插入数据
    map<int, string>::iterator iter;
    for(iter = mm.begin(); iter != mm.end(); iter++)
        cout<<iter->first<<' '<<iter->second<<endl;
    return 0;
}
```



## 3.3 获取/修改容器大小

```cpp
#include<iostream>
#include<map>
using namespace std;
int main(){
    map<int, string> mm;
    pair<map<int, string>::iterator, bool> Insert_Pair;
    Insert_Pair = mm.insert(pair<int,string>(0,"zero"));
    if(Insert_Pair.second == true)
        cout<<"Insert Successfully"<<endl;
    else
        cout<<"Insert Failure"<<endl;
    mm.insert(make_pair(1,"one"));
    mm.insert(map<int,string>::value_type(3,"three"));
    mm[4] = "four";
    int size = mm.size();  //获取map大小
    return 0;
}
```

## 3.4 访问元素

## 3.5 删除元素

```cpp
//iterator erase（iterator it);//通过一个条目对象删除
//iterator erase（iterator first，iterator last）//删除一个范围
//size_type erase(const Key&key);//通过关键字删除
//clear()就相当于enumMap.erase(enumMap.begin(),enumMap.end());
#include<iostream>
#include<map>
using namespace std;
int main(){
    map<int, string> mm;
    mm.insert(pair<int,string>(0,"zero"));
    mm.insert(make_pair(1,"one"));
    mm.insert(map<int,string>::value_type(3,"three"));
    mm[4] = "four";
    map<int, string>::iterator iter;
    iter = mm.find(3);
    mm.erase(iter);    //迭代器删除
    int n = mm.erase(0);  //关键字删除，成功返回1，失败返回0
    for(iter = mm.begin(); iter != mm.end(); iter++)
        cout<<iter->first<<' '<<iter->second<<endl;
    mm.erase(mm.begin(),mm.end()); //全部删除
    return 0;
}
```



## 3.6 遍历

```cpp
//第一种：应用前向迭代器
//第二种：应用反相迭代器
#include<iostream>
#include<map>
using namespace std;
int main(){
    map<int, string> mm;
    mm.insert(pair<int,string>(0,"zero"));  //插入pair数据
    mm.insert(make_pair(1,"one"));        //插入pair数据
    mm.insert(map<int,string>::value_type(3,"three"));  //插入value_type数据
    mm[4] = "four";                  //数组方式插入数据
    map<int, string>::iterator iter;
    for(iter = mm.begin(); iter != mm.end(); iter++)
        cout<<iter->first<<' '<<iter->second<<endl;
    map<int, string>::reverse_iterator riter;  
    for(riter = mapStudent.rbegin(); riter != mapStudent.rend(); riter++)  
        cout<<riter->first<<"  "<<riter->second<<endl; 
    return 0;
}
```



## 3.7 查找

```cpp
// 第一种：用count函数来判定关键字是否出现，其缺点是无法定位数据出现位置
// 第二种：用find函数来定位数据出现位置，它返回的一个迭代器，当数据出现时，它返回数据所在位置的迭代器，如果map中没有要查找的数据，它返回的迭代器等于end函数返回的迭代器
#include<iostream>
#include<map>
using namespace std;
int main(){
    map<int, string> mm;
    mm.insert(pair<int,string>(0,"zero"));
    mm.insert(make_pair(1,"one"));
    mm.insert(map<int,string>::value_type(3,"three"));
    mm[4] = "four";
    map<int, string>::iterator iter;
    iter = mm.find(4);
    if(iter != mm.end()){
        cout<<"find key:"<<iter->first<<" value:"<<iter->second<<endl;
    }else{
        cout<<"not find"<<endl;
    }
    for(iter = mm.begin(); iter != mm.end(); iter++)
        cout<<iter->first<<' '<<iter->second<<endl;
    return 0;
}
```

## 3.8 排序

map中的元素是自动按Key升序排序，所以不能对map用sort函数

STL中默认是采用小于号来排序的，以上代码在排序上是不存在任何问题的，因为上面的关键字是int 型，它本身支持小于号运算，在一些特殊情况，比如关键字是一个结构体，涉及到排序就会出现问题，因为它没有小于号操作，insert等函数在编译的时候过 不去；需要重载小于号

# 4. unordered_map

https://www.cnblogs.com/langyao/p/8823092.html

C++ 11标准中加入了unordered系列的容器。unordered_map记录元素的hash值，根据hash值判断元素是否相同,即unordered_map内部元素是无序的。

map中的元素是按照二叉搜索树存储（用红黑树实现），进行中序遍历会得到有序遍历。所以使用时map的key需要定义operator<

而unordered_map需要定义hash_value函数并且重载operator==

unordered_map编译时gxx需要添加编译选项：--std=c++11

# 5. queue

## 5.1 创建

```cpp
queue<int> mqueue;
queue<int> mqueue1{mqueue};
```



## 5.2 增加/插入数据

```cpp
queue<int> mqueue;
mqueue.push(1);
mqueue.emplace(2);  //可以避免对象的拷贝，重复调用构造函数
```



## 5.3 获取/修改容器大小

```cpp
queue<int> mqueue;
mqueue.push(1);
mqueue.emplace(2);
mqueue.size();
mqueue.empty();  //判断是否为空
```



## 5.4 访问元素

```cpp
mqueue.front();  //返回 queue 中第一个元素的引用
mqueue.back();  //返回 queue 中最后一个元素的引用
```



## 5.5 删除元素

```cpp
mqueue.pop();
```



## 5.6 遍历

和 stack 一样，queue 也没有迭代器。访问元素的唯一方式是遍历容器内容，并移除访问过的每一个元素

## 5.7 查找





# 6. deque

deque两端都能够快速插入和删除元素

Deque的操作函数和vector操作函数基本一模一样,duque的各项操作只有以下几点和vector不同:

1. deque不提供容量操作( capacity()、reserve() )
2. deque提供push_front()、pop_front()函数直接操作头部

deque元素是分布在一段段连续空间上，因此deque具有如下特点：

1、支持随机访问，即支持[]以及at()，但是性能没有vector好。

2、可以在内部进行插入和删除操作，但性能不及list。

 由于deque在性能上并不是最高效的，有时候对deque元素进行排序，更高效的做法是，将deque的元素移到到vector再进行排序，然后在移到回来。

## 6.1 创建

```cpp
deque<int> mqueue;
deque<int>  d(10);  //创建容量为10的deque
deque<int>  d2(6,8); //容量为6，所有元素初始化为8
int ar[5]={1,2,3,4,5};   //使用数组的一个区间初始化
deque<int>  d(ar,ar+5);
vector<double> vd{0.1,0.2,.05,.07,0.9};  //使用vector的一个区间初始化
deque<double>  d2(vd.begin()+1,vd.end());
deque<int> mqueue1{mqueue};  //使用另一个deque初始化
deque<int>  d2({1,2,3,4,5,6,7});  //初始化列表进行初始化

```



## 6.2 增加/插入数据

```cpp
deque<int> mqueue;
mqueue.push(1);
mqueue.emplace_front(2);  //可以避免对象的拷贝，重复调用构造函数
mqueue.emplace_back(2);  //可以避免对象的拷贝，重复调用构造函数
```



## 6.3 获取/修改容器大小

```cpp
deque<int> mqueue;
mqueue.push(1);
mqueue.emplace_front(2);
mqueue.size();
mqueue.empty();  //判断是否为空
```



## 6.4 访问元素

```cpp
mqueue.front();  //返回 queue 中第一个元素的引用
mqueue.back();  //返回 queue 中最后一个元素的引用
```



## 6.5 删除元素

```cpp
mqueue.pop_front();
mqueue.pop_end();
```



## 6.6 遍历

```cpp
for (std::deque<int>::iterator it = dq.begin(); it!=dq.end(); ++it)
    std::cout << ' ' << *it;
```



## 6.7 查找

# 7. stack

## 7.1  创建

```cpp
//stack<int> s1 = {1,2,3,4,5};   //error    stack不可以用一组数直接初始化
//stack<int> s2(10);             //error    stack不可以预先分配空间
stack<int> s3;

vector<int> v1 = {1,2,3,4,5};       // 1,2,3,4,5依此入栈
stack<int, vector<int>> s4(v1);

list<int> l1 = {1,2,3,4,5};
stack<int, list<int>> s5(l1);

deque<int> d1 = {1,2,3,4,5};
stack<int, deque<int>> s6(d1);
stack<int> s7(d1);                  //用deque 为 stack  初始化时 deque可省  因为stack是基于deque, 默认以deque方式构造
```



## 7.2 增加/插入数据

```cpp
mstack.push(333);
mstach.emplace(333);
```



## 7.3 获取/修改容器大小

```cpp
mstack.size();
mstack.empty();
```



## 7.4 访问元素

```cpp
mstack.top();
```



## 7.5 删除元素

```cpp
mstack.pop();
```



## 7.6 遍历

stack 遍历需要将所有元素出栈

```cpp
#include<iostream>
#include<stack>
#include<deque>
using namespace std;
int main(){
    deque<int> q1{1,2,3,4,5};
    stack<int> s(q1);
    while(!s.empty()){
        cout<<s.top()<<" ";
        s.pop();
    }
    cout<<endl;
    return 0;
}
```

# 8. priority_queue（堆）

和`queue`不同的就在于我们可以自定义其中数据的优先级, 让优先级高的排在队列前面,优先出队

优先队列具有队列的所有特性，包括基本操作，只是在这基础上添加了内部的一个排序，它本质是一个**堆**实现的

## 8.1 创建

```cpp
// 定义 priority_queue<Type, Container, Functional>
// Type 就是数据类型，Container 就是容器类型（Container必须是用数组实现的容器，比如vector,deque等等，但不能用 list。STL里面默认用的是vector），
// Functional 就是比较的方式，当需要用自定义的数据类型时才需要传入这三个参数，使用基本数据类型时，只需要传入数据类型，
// 默认是大顶堆

//升序队列;小顶堆
priority_queue <int,vector<int>,greater<int> > q;
//降序队列；大顶堆
priority_queue <int,vector<int>,less<int> >q;
```

```cpp
//pari的比较，先比较第一个元素，第一个相等比较第二个
#include <iostream>
#include <queue>
#include <vector>
using namespace std;
int main() 
{
    priority_queue<pair<int, int> > a;
    pair<int, int> b(1, 2);
    pair<int, int> c(1, 3);
    pair<int, int> d(2, 5);
    a.push(d);
    a.push(c);
    a.push(b);
    while (!a.empty()) 
    {
        cout << a.top().first << ' ' << a.top().second << '\n';
        a.pop();
    }
}
```

```cpp
//自定义类型
#include <iostream>
#include <queue>
using namespace std;

//方法1
struct tmp1 //运算符重载<
{
    int x;
    tmp1(int a) {x = a;}
    bool operator<(const tmp1& a) const
    {
        return x < a.x; //大顶堆
    }
};

//方法2
struct tmp2 //重写仿函数
{
    bool operator() (tmp1 a, tmp1 b) 
    {
        return a.x < b.x; //大顶堆
    }
};

int main() 
{
    tmp1 a(1);
    tmp1 b(2);
    tmp1 c(3);
    priority_queue<tmp1> d;
    d.push(b);
    d.push(c);
    d.push(a);
    while (!d.empty()) 
    {
        cout << d.top().x << '\n';
        d.pop();
    }
    cout << endl;

    priority_queue<tmp1, vector<tmp1>, tmp2> f;
    f.push(c);
    f.push(b);
    f.push(a);
    while (!f.empty()) 
    {
        cout << f.top().x << '\n';
        f.pop();
    }
}
```

## 8.2 增加/插入数据

```cpp
priority_queue<int> mqueue;
mqueue.push(1);
mqueue.emplace(2);  //可以避免对象的拷贝，重复调用构造函数
```



## 8.3 获取/修改容器大小

```cpp
priority_queue<int> mqueue;
mqueue.push(1);
mqueue.emplace(2);
mqueue.size();
mqueue.empty();  //判断是否为空
```



## 8.4 访问元素

```cpp
mqueue.top();  //返回 queue中第一个元素，即最大/最小的元素
```



## 8.5 删除元素

```cpp
mqueue.pop();
```



## 8.6 遍历

和 stack 一样，queue 也没有迭代器。访问元素的唯一方式是遍历容器内容，并移除访问过的每一个元素
## vector遍历方式
	//第一种遍历方式，下标
	cout << "第一种遍历方式，下标访问" << endl;
	for (int i = 0; i<m_testPoint.size(); i++ )
	{
		cout << m_testPoint[i].x << "	" << m_testPoint[i].y << endl;
	}
	
	//第二种遍历方式，迭代器
	cout << "第二种遍历方式，迭代器访问" << endl;
	for (vector<Point>::iterator iter = m_testPoint.begin(); iter != m_testPoint.end(); iter++)
	{
		cout << (*iter).x << "	" << (*iter).y << endl;
	}
	
	//第三种遍历方式，auto关键字
	cout << "C++11,第三种遍历方式，auto关键字" << endl;
	for (auto iter = m_testPoint.begin(); iter != m_testPoint.end(); iter++)
	{
		cout << (*iter).x << "	" << (*iter).y << endl;
	}
	 
	//第四种遍历方式，auto关键字的另一种方式
	cout << "C++11,第四种遍历方式，auto关键字" << endl;
	for (auto i : m_testPoint)
	{
		cout << i.x << "	" << i.y << endl;
	}

## vector拷贝

```cpp
vector<int> input({ 1,2,3,4,5 });
cout<<input.size();
int arr[5];
std::copy(input.begin(), input.begin()+5, arr);
for (int i = 0; i < 5; i++) {
	cout<<arr[i]<<endl;
}

vector<int> input({ 1,2,3,4,5 });
vector<int> input2;
input2 = { input.begin(), input.begin() + 5 };
for (int i = 0; i < input2.size(); i++) {
	cout << input2[i];
}
```

## vector求和

```
int arr [] = {10,20,30,40,50}
vector<int> va(&arr[0],&arr[5])
int sum = accumulate(va.begin(),va.end(),0 )
```

> 说明：
>
> T accumulate( InputIt first, InputIt last, T init );
>
> T accumulate( InputIt first, InputIt last, T init,BinaryOperation op );
>
> accumulate默认返回的是int类型，操作符默认是‘+’;当sum溢出时，将init类型改为long，则返回long类型

```c++
#include <iostream>
#include <vector>
#include <numeric>
#include <string>
#include <functional>
 
int main()
{
    std::vector<int> v{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int sum = std::accumulate(v.begin(), v.end(), 0);
    int product = std::accumulate(v.begin(), v.end(), 1, std::multiplies<int>());
    auto dash_fold = [](std::string a, int b) {
                         return std::move(a) + '-' + std::to_string(b);
                     };
    std::string s = std::accumulate(std::next(v.begin()), v.end(),
                                    std::to_string(v[0]), // 用首元素开始
                                    dash_fold);
    // 使用逆向迭代器右折叠
    std::string rs = std::accumulate(std::next(v.rbegin()), v.rend(),
                                     std::to_string(v.back()), // 用首元素开始
                                     dash_fold);
    std::cout << "sum: " << sum << '\n'
              << "product: " << product << '\n'
              << "dash-separated string: " << s << '\n'
              << "dash-separated string (right-folded): " << rs << '\n';
}

sum: 55
product: 3628800
dash-separated string: 1-2-3-4-5-6-7-8-9-10
dash-separated string (right-folded): 10-9-8-7-6-5-4-3-2-1
```





## c_str()

c_str()函数返回一个指向C字符串的指针常量，内容与string相同

是为了与C语言兼容，C语言中没有string类型

```cpp
char c[20];
string s="1234";
strcpy(c,s.c_str());
```

## char和int之间转换

```cpp
#define KALDI_SWAP8(a) { \
  int t = (reinterpret_cast<char*>(&a))[0];\
          (reinterpret_cast<char*>(&a))[0]=(reinterpret_cast<char*>(&a))[7];\
          (reinterpret_cast<char*>(&a))[7]=t;\
      t = (reinterpret_cast<char*>(&a))[1];\
          (reinterpret_cast<char*>(&a))[1]=(reinterpret_cast<char*>(&a))[6];\
          (reinterpret_cast<char*>(&a))[6]=t;\
      t = (reinterpret_cast<char*>(&a))[2];\
          (reinterpret_cast<char*>(&a))[2]=(reinterpret_cast<char*>(&a))[5];\
          (reinterpret_cast<char*>(&a))[5]=t;\
      t = (reinterpret_cast<char*>(&a))[3];\
          (reinterpret_cast<char*>(&a))[3]=(reinterpret_cast<char*>(&a))[4];\
          (reinterpret_cast<char*>(&a))[4]=t;}
#define KALDI_SWAP4(a) { \
  int t = (reinterpret_cast<char*>(&a))[0];\
          (reinterpret_cast<char*>(&a))[0]=(reinterpret_cast<char*>(&a))[3];\
          (reinterpret_cast<char*>(&a))[3]=t;\
      t = (reinterpret_cast<char*>(&a))[1];\
          (reinterpret_cast<char*>(&a))[1]=(reinterpret_cast<char*>(&a))[2];\
          (reinterpret_cast<char*>(&a))[2]=t;}
#define KALDI_SWAP2(a) { \
  int t = (reinterpret_cast<char*>(&a))[0];\
          (reinterpret_cast<char*>(&a))[0]=(reinterpret_cast<char*>(&a))[1];\
          (reinterpret_cast<char*>(&a))[1]=t;}
uint32 ReadUint32() {
    union {
      char result[4];
      uint32 ans;
    } u;
    is.read(u.result, 4);
    if (swap)
      KALDI_SWAP4(u.result);
    if (is.fail())
      KALDI_ERR << "WaveData: unexpected end of file or read error";
    return u.ans;
}
uint16 ReadUint16() {
    union {
      char result[2];
      int16 ans;
    } u;
    is.read(u.result, 2);
    if (swap)
      KALDI_SWAP2(u.result);
    if (is.fail())
      KALDI_ERR << "WaveData: unexpected end of file or read error";
    return u.ans;
}
static void WriteUint32(std::ostream &os, int32 i) {
  union {
    char buf[4];
    int i;
  } u;
  u.i = i;
#ifdef __BIG_ENDIAN__
  KALDI_SWAP4(u.buf);
#endif
  os.write(u.buf, 4);
  if (os.fail())
    KALDI_ERR << "WaveData: error writing to stream.";
}
static void WriteUint16(std::ostream &os, int16 i) {
  union {
    char buf[2];
    int16 i;
  } u;
  u.i = i;
#ifdef __BIG_ENDIAN__
  KALDI_SWAP2(u.buf);
#endif
  os.write(u.buf, 2);
  if (os.fail())
    KALDI_ERR << "WaveData: error writing to stream.";
}
```

## 时间统计

```cpp
#include<chrono>
auto begin_t = std::chrono::steady_clock::now();
auto finish_t = std::chrono::steady_clock::now();
double timecost = std::chrono::duration<double, std::milli>(finish_t - begin_t).count();
```


# C++使用笔记

## 继承和派生

```text
继承和派生是同一概念。
派生类名::派生类名(参数表):基类名1(参数表),基类名2(参数表)
{
    本类成员初始化赋值语句
}

派生类名::派生类名(参数表):基类名1(参数表),基类名2(参数表),新增成员对象的初始化
{
    本类成员初始化赋值语句
}
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

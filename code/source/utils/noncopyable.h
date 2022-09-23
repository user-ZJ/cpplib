#pragma once

// 不可拷贝的基类，继承该类的所有类均不可拷贝
namespace BASE_NAMESPACE{


// 如果基类中的默认构造函数、拷贝构造函数、拷贝赋值运算符或析构函数是被删除的函数或者不可访问，则派生类中对应的成员将是被删除的，
// 原因是编译器不能使用基类成员来执行派生类对象基类部分的构造、赋值或销毁操作。

class noncopyable
{
 public:
  noncopyable(const noncopyable&) = delete;
  void operator=(const noncopyable&) = delete;

 protected:
  noncopyable() = default;
  ~noncopyable() = default;
};

};
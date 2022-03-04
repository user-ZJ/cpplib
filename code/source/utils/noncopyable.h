#pragma once

// 不可拷贝的基类，继承该类的所有类均不可拷贝
namespace BASE_NAMESPACE{

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
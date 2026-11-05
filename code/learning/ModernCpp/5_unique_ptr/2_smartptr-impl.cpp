
#include <cstdlib>
#include <cassert>
#include <utility>
#include <iostream>

using namespace std;

template <typename T>
class SmartPtr {
public:
    explicit SmartPtr(T* p = nullptr)
        : p_(p) {
        }

    ~SmartPtr() { 
        delete p_;
    }
    void release() { 
        delete p_;
        p_ = nullptr; 
    }
   
    T* operator->() { return p_; }
    const T* operator->() const { return p_; }

    T& operator*() { return *p_; }
    const T& operator*() const { return *p_; }
	
private:
    T* p_;
    SmartPtr(const SmartPtr&) = delete;
    SmartPtr& operator=(const SmartPtr&) = delete;
};

class MyClass{

public:
    MyClass(){
        cout<<"MyClass 初始化"<<endl;
    }


    ~MyClass(){
        cout<<"MyClass 析构"<<endl;
    }


};

int main() {
    
    SmartPtr p(new MyClass());
}

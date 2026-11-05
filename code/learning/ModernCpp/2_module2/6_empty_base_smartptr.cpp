

#include <cstdlib>
#include <cassert>
#include <utility>
#include <iostream>

template <typename T>
struct DeleteByOperator {
    void operator()(T* p) const { 
        delete p;
    }
};

template <typename T>
struct DeleteByFree {
    void operator()(T* p) const {
        p->~T();
        free(p);
    }
};

template <typename T>
struct DeleteDestructorOnly {
    void operator()(T* p) const {
        p->~T();
    }
};

template <typename DeletionPolicy>
class SmartPtr { // 空基类优化
    DeletionPolicy deleter;
    
};



template <typename T>
class SmartPtr1 : private T { // 空基类优化

};

int main(){
    return 0;
}


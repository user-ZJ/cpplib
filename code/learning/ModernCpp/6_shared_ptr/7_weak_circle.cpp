#include <iostream>
#include <memory>
 
using namespace std;
#include <memory>
#include <iostream>


struct BClass;
struct CClass;

struct AClass
{
    shared_ptr<BClass> pb;
    ~AClass() { std::cout << "~AClass()\n"; }
};


struct BClass
{
    weak_ptr<CClass> pc;
    ~BClass() { std::cout << "~BClass()\n"; }
};

struct CClass
{
    shared_ptr<AClass> pa;
    ~CClass() { std::cout << "~CClass()\n"; }
};

int main()
{
    {
        shared_ptr<AClass> a = std::make_shared<AClass>();
        shared_ptr<BClass> b = std::make_shared<BClass>();
        shared_ptr<CClass> c = std::make_shared<CClass>();

        // 循环引用
        a->pb = b;
        b->pc = c; //弱引用 不算引用计数
        c->pa = a;
        // c 释放

        //c.reset();
        std::cout << "计数: " << a.use_count() << "\n"; 
        std::cout << "计数: " << b.use_count() << "\n";
        std::cout << "计数: " << c.use_count() << "\n";  
        
      
    }



 

    // a, b 仍然相互持有
}


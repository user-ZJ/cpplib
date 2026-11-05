#include <iostream>
using namespace std;

#include <iostream>
#include <memory>

struct Widget {
    int data;
    Widget(int d):data(d){
        cout<<"ctor"<<endl;
    }
    ~Widget()
    {
        cout<<"dtor"<<endl;
    }
};

weak_ptr<Widget> wptr;

void check() {
    cout << "wptr 引用计数 " << wptr.use_count() << endl;
    
    if ( !wptr.expired()) {

        shared_ptr<Widget> sptr = wptr.lock();

        cout << "wptr 引用计数 " << wptr.use_count() << endl;
        cout << "sptr 引用计数 " << sptr.use_count() << endl;
        
        
        cout << "当前值 " << sptr->data << '\n';
    }
    else {
        cout << "wptr 已销毁\n";
    }

    cout << "wptr 引用计数 " << wptr.use_count() << endl;
        
}
int main() {
    {
        shared_ptr<Widget> s = make_shared<Widget>(42);
        wptr = s;
        check();
    } // s 销毁
    cout<<"s销毁后......"<<endl;
    check();
}


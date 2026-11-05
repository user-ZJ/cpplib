#include <memory>
#include <iostream>

using namespace std;


class Widget{};

// 推荐：仅仅使用这个 widget，不表达任何所有权
void process1(Widget*){}
void process2(const Widget&){}



//---------unique_ptr：独占所有权------------

// 推荐，也常用：获取 widget 的所有权
void process3(unique_ptr<Widget>){}
void process3(unique_ptr<Widget>&& ){}

// 可行，不常用：打算重新指向别的对象
void process4(unique_ptr<Widget>&){}

// 不推荐： 通常不是想要的
void process5(const unique_ptr<Widget>&){}



//--------- shared_ptr ：共享所有权------------

// 推荐，常用：成为共享所有权之一
void share(shared_ptr<Widget>){}
 
// 可行，不常用：可能保留引用计数 
void may_share(const shared_ptr<Widget>&){}

// 可行，不常用：打算重新指向别的对象
void reseat(shared_ptr<Widget>&){}

int main(){
    return 0;
}

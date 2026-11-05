#include <memory>
#include <iostream>

using namespace std;

class Widget{};

// 推荐：仅仅使用这个 widget，不表达任何所有权
void process1(Widget *w){}
void process2(const Widget&){}




// 推荐，也常用：获取 widget 的所有权
void process3(unique_ptr<Widget>){}







// 可行，不常用：打算重新指向别的对象
void process4(unique_ptr<Widget>&){}

// 不推荐： 通常不是想要的
void process5(const unique_ptr<Widget>&){}

int main(){
unique_ptr<Widget> u=make_unique<Widget>();
//process1(u.get());
process2(*u);
}
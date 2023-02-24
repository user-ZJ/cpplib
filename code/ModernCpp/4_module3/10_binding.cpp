

class Widget{};

Widget getWidget(){
    Widget w;
    return w;
}

void f1(Widget&){}
void f2(const Widget&){}
void f3(Widget&&){}
void f4(const Widget&&){}

int main(){
Widget w1;
f1(w1);//OK
f2(w1);//OK
// f3(w1);//ERROR!
// f4(w1);//ERROR!

const Widget w2;
// f1(w2);//ERROR
f2(w2);//OK
// f3(w2);//ERROR!
// f4(w2);//ERROR!

// f1(getWidget()); //ERROR
f2(getWidget()); //OK 延长了临时对象的生命周期
f3(getWidget()); //OK
f4(getWidget()); //OK

const auto&&  temp=getWidget();
// f1(temp); //ERROR
f2(temp); // OK
// f3(temp); //ERROR
// f4(temp); // OK 

}
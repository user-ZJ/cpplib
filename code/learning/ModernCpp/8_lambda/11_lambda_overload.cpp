
#include <iostream>
#include <memory>
#include <vector>
#include <variant>

using namespace std;


/*
struct overloaded : Lam1,Lam2,Lam3,Lam4 { 

    using Lam1::operator();
    using Lam2::operator();
    using Lam3::operator();
    using Lam4::operator();
    
};




struct Lam1{
    operator()(int data){
        cout <<"int:"<<d<<endl;
    }
};

struct Lam2{
    operator()(double data){
        cout <<"double:"<< d;
    }
};

struct Lam3{
    operator()(bool data){

    }
};

struct Lam4{
    operator()(string data){

    }
};


*/



template<class... Ts> 
struct overloaded : Ts... { 
    using Ts::operator()...;  //继承父类所有operator() 操作符
};

template<class... Ts> 
overloaded(Ts...) -> overloaded<Ts...>; //自定义模板推导


struct my_overloaded { 

    void operator()(int data){
        cout <<"int:"<<data<<endl;
    }
    void operator()(double data){
        cout <<"double:"<< data;
    }
    void operator()(bool data){
        cout <<"bool: "<< data;
    }

    void operator()(string data){
        cout <<"string: "<< data;
    }
};



int main(){


    std::variant<int,double ,bool, string> data;
    data="hello"s;
    
    
    auto lams=overloaded{
                        [](int d) { cout <<"int:"<<d<<endl; } ,
                        [](double d) { cout <<"double:"<< d; },
                        [](bool d) { cout <<"bool: "<< d; },  
                        [](string d) { cout <<"string: "<< d; }     
    }; //==> overloaded<Lam1,Lam2,Lam3,Lam4> { l1,l2,l3,l4};

    //my_overloaded lams;

    /*
    Lam1 l1;
    Lam2 l2;
    Lam3 l3;
    Lam4 l4;
    overloaded<Lam1,Lam2,Lam3,Lam4> ol{l1,l2,l3,l4};*/

   
    std::visit(lams, data);

    cout<<endl;


}
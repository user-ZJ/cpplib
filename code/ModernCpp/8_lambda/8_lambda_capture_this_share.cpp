#include <functional>
#include <iostream>
#include <vector>
#include <memory>

using namespace std;

struct TCPSession: public enable_shared_from_this<TCPSession>
{

    int data;

    TCPSession(int d):data(d)
    {

    }

    TCPSession(const TCPSession& rhs):data(rhs.data){
        cout<<"TCPSession copy ctor"<<endl;
    }

    void process() const{
        cout<<"process:"<<data<<endl;

    }


     auto getLambda(){
        
        //shared_ptr<TCPSession> self=make_shared<TCPSession>(this); 

        shared_ptr<TCPSession> self=shared_from_this();

        auto lam=[this,self]() {

            data++;
            process();

        }; 

        return lam;
    }

    ~TCPSession(){
        cout<<"~TCPSession"<<endl;
    }
};


std::function<void(void)> process()
{
    shared_ptr<TCPSession> tsObj=make_shared<TCPSession>(100);
        
    auto lambda=tsObj->getLambda();

    cout<<"引用计数："<<tsObj.use_count()<<endl;


    return lambda;
}


int main(){

    std::function<void(void)> func=process();

    func();


}
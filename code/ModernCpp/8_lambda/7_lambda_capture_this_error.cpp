#include <functional>
#include <iostream>
#include <vector>

using namespace std;

struct TCPSession
{
    int data;

    TCPSession(int d):data(d){  }

    TCPSession(const TCPSession& rhs):data(rhs.data){
        cout<<"TCPSession copy ctor"<<endl;
    }

     ~TCPSession(){
        cout<<"~TCPSession"<<endl;
    }

    void process() const{
        cout<<"process:"<<data<<endl;

    }
 
    auto getLambda(){
        
        auto lam=[ this]() {
            
            data++;
            process();
        };
        return lam;
    }
};

std::function<void(void)> process()
{
    TCPSession tsObj(100);
    auto lambda=tsObj.getLambda();
    return lambda;
}

int main(){

    std::function<void(void)> func=process();

    func();
}


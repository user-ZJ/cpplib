#include <iostream>
#include <vector>
#include <map>
#include <list>
#include <array>
#include <memory>
using namespace std;


constinit int ONEK = 1024;
constexpr int ONEK_PR = 1024;

//constinit int ONEM=ONEK*ONEK; //错误! 运行时初始化
constinit int ONEM_PR=ONEK_PR*ONEK_PR; //OK

int getNextNumber() {
    static constinit int minNumber = 0; //可static
    return ++minNumber;
}

struct Widget {
    inline static constinit long MaxSize = sizeof(int) * ONEK_PR;
    inline static thread_local constinit int numCalls = 0;

    static constexpr long MinSize = sizeof(int) ;
};



constexpr std::array<int, 8> create_prime() {
    return {1, 2, 3, 5, 7, 11, 13, 17};
}

constinit auto global_prime = create_prime();

constinit std::pair p1{1024, "K"}; //构造器支持编译时初始化
constinit std::unique_ptr<int> upInteger;
constinit std::shared_ptr<int> spInteger;





int main(){

    cout<<ONEK<<endl;
    ONEK*=1024; //运行时可更改
    
    global_prime[0]=19;//运行时可更改

    Widget::MaxSize++; //运行时可更改
    cout<<Widget::MaxSize<<endl;
    cout<<p1.first<<","<<p1.second<<endl;

   

}
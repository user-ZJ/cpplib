# boost log

## 简单示例

```cpp
#include <iostream>
#include <thread>

void thread_function()
{
    for(int i = 0; i < 10000; i++);
        std::cout<<"thread function Executing"<<std::endl;
}

int main()
{

    std::thread threadObj(thread_function);
    for(int i = 0; i < 10000; i++);
        std::cout<<"Display From MainThread"<<std::endl;
    threadObj.join();
    std::cout<<"Exit of Main function"<<std::endl;
    return 0;
}
```

```shell
g++ boost_test.cpp -o boost_test -L/root/boost_1_75_0/stage/lib/ -lboost_log -lboost_log_setup -lpthread
```


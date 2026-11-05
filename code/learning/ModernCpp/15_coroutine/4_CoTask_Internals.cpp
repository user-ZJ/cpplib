#include <coroutine>
#include <exception>  
#include <iostream>

using namespace std;
//using namespace std::experimental;


struct CoTask {

    struct promise_type { 
        CoTask get_return_object() { 
            return Co_Handle::from_promise(*this); 
        }    
        suspend_always initial_suspend() { return {}; }        
        suspend_always final_suspend() noexcept { return {}; }
        void unhandled_exception() {   terminate();  }
        
        void return_void() noexcept {} 
    };

    using Co_Handle=coroutine_handle<promise_type>;

    CoTask(Co_Handle  handle) : 
        co_handle(handle) {}

    CoTask(CoTask&& c) noexcept
        : co_handle{std::move(c.co_handle)} {
        c.co_handle = nullptr;
    }

    CoTask& operator=(CoTask&& c) noexcept {
        if (this == &c)
        {
        return *this;
        }

        if (co_handle) {
        co_handle.destroy(); 
        }
        co_handle = std::move(c.co_handle); 
        c.co_handle = nullptr; 
        
        return *this;
    }

    CoTask(const CoTask& c) =delete;
    CoTask& operator=(const CoTask& c) =delete;

     // resume 协程
    bool resume()  {
        if (!co_handle || co_handle.done()) {
        return false;   
        }
        co_handle.resume();   
        return !co_handle.done();
    }

    ~CoTask() { co_handle.destroy(); }



    Co_Handle   co_handle;
};



CoTask process() {

    cout << "Hello "<<endl;
    co_await suspend_always{};
    cout << "Cpp "<<endl;
    co_await suspend_always{};
    cout << "World!" <<endl;
}

/*

struct __CoTask_ctx {
    CoTask::promise_type _promise;
    // 传给coroutine的实参
    // 局部变量
    // 当前挂起点 
};

//简化版
CoTask process() {

    //协程上下文状态
    __CoTask_ctx* __context = new __CoTask_ctx{};//分配在堆上
    CoTask __return = __context->_promise.get_return_object();
    co_await __context->_promise.initial_suspend();

    try
    {
        cout << "Hello "<<endl;
        co_await suspend_always{};
        cout << "Cpp "<<endl;
        co_await suspend_always{};
        cout << "World!" <<endl;
    }
    catch(...)
    {
        __context->_promise.unhandled_exception();
    }

__final_suspend_label:
    co_await __context->_promise.final_suspend();
    delete __context;
    return __return;
}




//加强版
CoTask process() {
    __CoTask_ctx* __context = new __CoTask_ctx{};
    auto __return = __context->_promise.get_return_object();
    {
        auto&& awaiter = std::suspend_always{};
        if (!awaiter.await_ready()) {
            awaiter.await_suspend(std::coroutine_handle<> p); 
        }
        awaiter.await_resume();
    }

    std::cout << "Hello ";
    {
        auto&& awaiter = std::suspend_always{};
        if (!awaiter.await_ready()) {
            awaiter.await_suspend(std::coroutine_handle<> p); 
        }
        awaiter.await_resume();
    }
    std::cout << "Cpp ";
    {
        auto&& awaiter = std::suspend_always{};
        if (!awaiter.await_ready()) {
            awaiter.await_suspend(std::coroutine_handle<> p); 
        }
        awaiter.await_resume();
    }
    std::cout << "World!" << std::endl;

__final_suspend_label:
    {
        auto&& awaiter = std::suspend_always{};
        if (!awaiter.await_ready()) {
            awaiter.await_suspend(std::coroutine_handle<> p); 
        }
        awaiter.await_resume();
    }

    delete __context;
    return __return;
}
*/

int main() {
    CoTask coTask = process();

    coTask.co_handle.resume();
    coTask.co_handle.resume(); 
    coTask.co_handle.resume(); 
   
}

#include <concepts>
#include <exception>
#include <iostream>

#include <coroutine>

using namespace std;
//using namespace std::experimental;


int alloc_times = 0; 
int dealloc_times = 0; 
int allocated = 0; 



void* operator new(size_t size)  { 
  void* p = std::malloc(size); 
  allocated+=size;
  alloc_times++;
  return p; 
} 
 
void operator delete(void* p) noexcept  { 
  dealloc_times++;
  return std::free(p); 
}

void* operator new[](size_t size)   {
  void* p = std::malloc(size); 
  allocated+=size;
  alloc_times++;

  return p; 
} 
void operator delete[](void* p) noexcept  { 
  dealloc_times++;
  return std::free(p); 
}


class CoTask {

public:
  struct promise_type {

    //协程不返回任何值
    void return_void() noexcept {} 

    CoTask get_return_object() { return CoHandle::from_promise(*this);}

    //初始化后的行为
    suspend_always initial_suspend() {  //初始化后立即挂起
      return suspend_always{}; 
    }
    suspend_always final_suspend() noexcept { //定义是否被最终挂起
      return suspend_always{}; 
    }
    void unhandled_exception() {   //协程内出现异常的处理逻辑
      terminate();  
    }

   
  };

  using CoHandle = coroutine_handle<promise_type>;


  CoTask(CoHandle h): co_handle{h} {         
  } 

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


  ~CoTask() {
    if (co_handle) {
      co_handle.destroy();      //销毁协程句柄
    }
  }

  // resume 协程
  bool resume()  {
    if (!co_handle || co_handle.done()) {
      return false;   
    }
    co_handle.resume();   
    return !co_handle.done();
  }

private:
  CoHandle co_handle;


};



CoTask counter()
{
   double data1, data2;

  for (int i = 0;; ++i) {
    
    cout << "counter: " << i << endl;
    co_await suspend_always();
  }
}

CoTask counter(double num)
{
  
  for (int i = 0;i<num; ++i) {
    
    cout << "counter: " << i << endl;
    co_await suspend_always();
  }
}

CoTask counter(double num, string text)
{
  
  for (int i = 0;i<num; ++i) {
    
    cout << text+ " : " << i << endl;
    co_await suspend_always();
  }
}



int main()
{

  
  // {
  //   CoTask cotask=counter();
  //   for (int i = 0; i < 5; ++i) {
  //     cotask.resume();
  //     cout << "back to main \n";
  //   }
  // }


  //  {
  //   CoTask cotask=counter(5);

  //   while(cotask.resume())
  //   {
  //     cout << "back to main \n";
  //   }
  // }
  

   {
    CoTask cotask=counter(5, "mark");

    while(cotask.resume())
    {
      cout << "back to main \n";
    }

    
   }

  cout<<"总分配："<< allocated<<" bytes, 分配次数："<<alloc_times<<" 释放次数："<<dealloc_times<<endl;
  

}

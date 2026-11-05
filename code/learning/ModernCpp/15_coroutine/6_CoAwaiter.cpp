#include <coroutine>
#include <exception>  
#include <iostream>

using namespace std;


class Awaiter {
 public:
  bool await_ready() const noexcept {
    cout << "   await_ready()"<<endl;
    return false; // 挂起    
  }

  void await_suspend(auto hdl) const noexcept {
    cout << "   await_suspend()"<<endl;
  }

  void await_resume() const noexcept {
    cout << "   await_resume()"<<endl;
  }
};


class CoTask {

 public:
  struct promise_type;
 private:
  using CoHandle = coroutine_handle<promise_type>;
  CoHandle co_handle;        

 public:
  struct promise_type {
    auto get_return_object() {      
      return CoTask{CoHandle::from_promise(*this)};
    }
    auto initial_suspend() {         
      return suspend_always{};  
    }
    void unhandled_exception() {   
      terminate();             
    }
    void return_void() {       
    }
    auto final_suspend() noexcept {  
      return suspend_always{}; 
    }
  };


  CoTask(auto h)
   : co_handle{h} {         
  }
  ~CoTask() {
    if (co_handle) {
      co_handle.destroy();    
    }
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

  bool resume()  {
    if (!co_handle || co_handle.done()) {
      return false;     
    }
    co_handle.resume();      
    return !co_handle.done();
  }
};

CoTask step(int max)
{
  cout << "  step start"<<endl;

  for (int val = 1; val <= max; ++val) {
    cout << "  step " << val << endl;
    co_await Awaiter{}; 
  }

  std::cout << "  step end"<<endl;
}

int main()
{

  cout << "main() started"<<endl;
  CoTask coTask = step(3);
  

  cout << "loop start"<<endl;
  while (coTask.resume()) {  
    cout << "main() continue"<<endl;
  }

  cout << "main() end"<<endl;
}

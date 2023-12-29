#include <iostream>
#include <vector>
#include <ranges>
#include <exception>  
#include <coroutine>

using namespace std;



template<typename T>
class CoResult {
 public:

  struct promise_type {
    T result{};        // co_return 返回值
    
    // 接受 co_return的值 
    void return_value(const auto& value) { 
      result = value;                     
    }

    CoResult get_return_object() { return CoHandle::from_promise(*this);}

    auto initial_suspend() { return suspend_always{}; }
    auto final_suspend() noexcept { return suspend_always{}; }
    void unhandled_exception() { terminate(); }

  };

  using CoHandle = coroutine_handle<promise_type>;


  CoResult(CoHandle h) : co_handle{h} { }
  ~CoResult() { 
      if (co_handle) 
        co_handle.destroy(); 
    }

  CoResult(const CoResult&) = delete;
  CoResult& operator=(const CoResult&) = delete;

 

  CoResult(CoResult&& c) noexcept
    : co_handle{std::move(c.co_handle)} {
    c.co_handle = nullptr;
  }

  CoResult& operator=(CoResult&& c) noexcept {
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

  bool resume()  {
    if (!co_handle || co_handle.done()) {
      return false;    
    }
    co_handle.resume();    
    return !co_handle.done();
  }

  // 获取co_return的值
  T getResult() const {
    return co_handle.promise().result;
  }

private:
  CoHandle co_handle;  
};


CoResult<double> co_sum(auto coll)
{
  double result = 0;
  for (const auto& elem : coll) {
    cout << "  process " << elem << '\n';
    result  += elem;
    co_await suspend_always(); 
  }
  co_return result ;  //__context->_promise.return_value(result);
}

int main()
{
  vector values{1, 2, 3, 4, 5};

  CoResult<double> task = co_sum(values);

  cout << "main() start"<<endl;
  while (task.resume()) {                
    cout << "main() continue"<<endl;
    cout << "main() result: " << task.getResult() <<endl;
  }

  cout << "main() result: " << task.getResult() <<endl;
}



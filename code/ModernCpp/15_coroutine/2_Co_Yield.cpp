#include <coroutine>
#include <exception>
#include <iostream>

using namespace std;
// using namespace std::experimental;

class CoValue {
 public:
  struct promise_type {
    int value = 0;  // co_yield 值

    // 接受 co_yield 值 , 然后挂起
    suspend_always yield_value(int val) {
      value = val;
      return suspend_always{};
    }

    auto get_return_object() {
      return CoHandle::from_promise(*this);
    }

    auto initial_suspend() {
      return suspend_always{};
    }
    void return_void() {}
    void unhandled_exception() {
      std::terminate();
    }
    auto final_suspend() noexcept {
      return suspend_always{};
    }
  };

  using CoHandle = coroutine_handle<promise_type>;

  CoValue(CoHandle h) : co_handle{h} {}

  ~CoValue() {
    if (co_handle) co_handle.destroy();
  }

  CoValue(const CoValue &) = delete;
  CoValue &operator=(const CoValue &) = delete;

  // 获取co_yield的值
  int getValue() const {
    return co_handle.promise().value;
  }

  bool resume() {
    if (!co_handle || co_handle.done()) { return false; }
    co_handle.resume();
    return !co_handle.done();
  }

 private:
  CoHandle co_handle;
};

CoValue create_value(int max) {
  for (int val = 1; val <= max; ++val) {
    cout << "create_value: " << val << ':' << max << endl;

    co_yield val;  // co_await __context->_promise.yield_value(val);
  }

  std::cout << "create_value end: " << max << endl;
}

int main() {
  cout << "main() start" << endl;
  CoValue co_value = create_value(3);
  cout << "main() continue" << endl;

  while (co_value.resume()) {
    int val = co_value.getValue();
    cout << "main : co_value=" << val << '\n';
  }

  cout << "main() end" << endl;
}

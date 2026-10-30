// 完整的惰性 Task：co_await 嵌套、co_return 取值、异常传回调用方。
// 协程在挂起点把句柄交给 Executor，只有 Executor::run 会 resume，避免把还在等待子任务的父协程再次拉起。

#include <coroutine>
#include <deque>
#include <exception>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

class Executor {
 public:
  void schedule(std::coroutine_handle<> handle) { ready_.push_back(handle); }

  void run() {
    while (!ready_.empty()) {
      const auto handle = ready_.front();
      ready_.pop_front();
      handle.resume();
    }
  }

 private:
  std::deque<std::coroutine_handle<>> ready_;
};

// 主动让出一次。await_suspend 返回 void，协程挂起，稍后由 Executor 恢复。
struct Step {
  Executor& executor;
  const char* point;

  bool await_ready() const noexcept { return false; }

  void await_suspend(std::coroutine_handle<> current) const {
    std::cout << "挂起 " << point << '\n';
    executor.schedule(current);
  }

  void await_resume() const { std::cout << "恢复 " << point << '\n'; }
};

template <typename T>
class Task {
 public:
  struct promise_type {
    std::optional<T> value;
    std::exception_ptr exception;
    std::coroutine_handle<> continuation;

    Task get_return_object() { return Task{handle_type::from_promise(*this)}; }

    std::suspend_always initial_suspend() noexcept { return {}; }

    void unhandled_exception() noexcept { exception = std::current_exception(); }

    void return_value(T result) { value = std::move(result); }

    // 子协程结束后直接把控制权交回父协程；根协程没有 continuation，回到 Executor。
    struct FinalAwaiter {
      bool await_ready() const noexcept { return false; }

      std::coroutine_handle<> await_suspend(std::coroutine_handle<promise_type> current) noexcept {
        if (current.promise().continuation) {
          return current.promise().continuation;
        }
        return std::noop_coroutine();
      }

      void await_resume() const noexcept {}
    };

    FinalAwaiter final_suspend() noexcept { return {}; }
  };

  using handle_type = std::coroutine_handle<promise_type>;

  explicit Task(handle_type handle) : handle_(handle) {}

  Task(const Task&) = delete;
  Task& operator=(const Task&) = delete;

  Task(Task&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}

  Task& operator=(Task&& other) noexcept {
    if (this != &other) {
      destroy();
      handle_ = std::exchange(other.handle_, nullptr);
    }
    return *this;
  }

  ~Task() { destroy(); }

  handle_type handle() const { return handle_; }

  bool await_ready() const noexcept { return handle_.done(); }

  std::coroutine_handle<> await_suspend(std::coroutine_handle<> caller) noexcept {
    handle_.promise().continuation = caller;
    return handle_;
  }

  T await_resume() {
    auto& promise = handle_.promise();
    if (promise.exception) {
      std::rethrow_exception(promise.exception);
    }
    return *promise.value;
  }

  T result() const {
    if (!handle_ || !handle_.done()) {
      throw std::logic_error("task is not finished");
    }
    auto& promise = handle_.promise();
    if (promise.exception) {
      std::rethrow_exception(promise.exception);
    }
    return *promise.value;
  }

 private:
  void destroy() {
    if (handle_) {
      handle_.destroy();
      handle_ = nullptr;
    }
  }

  handle_type handle_{};
};

Task<int> add_one(Executor& executor, int value) {
  co_await Step{executor, "add_one"};
  co_return value + 1;
}

Task<int> compute(Executor& executor) {
  const int left = co_await add_one(executor, 19);
  const int right = co_await add_one(executor, 21);
  co_return left + right;
}

Task<int> fail(Executor& executor) {
  co_await Step{executor, "fail"};
  throw std::runtime_error("coroutine failed");
}

Task<int> propagate(Executor& executor) {
  co_return co_await fail(executor);
}

int main() {
  Executor executor;

  auto task = compute(executor);
  executor.schedule(task.handle());
  executor.run();
  std::cout << "结果 " << task.result() << '\n';

  auto broken = propagate(executor);
  executor.schedule(broken.handle());
  executor.run();
  try {
    broken.result();
  } catch (const std::exception& error) {
    std::cout << "异常 " << error.what() << '\n';
  }
}

// 不挂起等待外部事件时，调用处只需要范围 for 或 sync_wait。
// 协程类型本身仍要写一次，之后业务函数不用接触 promise 和调度器。

#include <coroutine>
#include <exception>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>
#include <utility>

template <typename T>
class Generator {
 public:
  struct promise_type {
    std::optional<T> current;

    Generator get_return_object() { return Generator{handle_type::from_promise(*this)}; }
    std::suspend_always initial_suspend() noexcept { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }
    std::suspend_always yield_value(T value) {
      current = std::move(value);
      return {};
    }
    void return_void() noexcept {}
    void unhandled_exception() noexcept { std::terminate(); }
  };

  using handle_type = std::coroutine_handle<promise_type>;

  explicit Generator(handle_type handle) : handle_(handle) {}
  Generator(const Generator&) = delete;
  Generator& operator=(const Generator&) = delete;
  Generator(Generator&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
  Generator& operator=(Generator&& other) noexcept {
    if (this != &other) {
      destroy();
      handle_ = std::exchange(other.handle_, nullptr);
    }
    return *this;
  }
  ~Generator() { destroy(); }

  class iterator {
   public:
    explicit iterator(handle_type handle) : handle_(handle) {}

    iterator& operator++() {
      handle_.resume();
      return *this;
    }
    const T& operator*() const { return *handle_.promise().current; }
    bool operator==(std::default_sentinel_t) const { return handle_.done(); }

   private:
    handle_type handle_;
  };

  iterator begin() {
    if (handle_) {
      handle_.resume();
    }
    return iterator{handle_};
  }
  std::default_sentinel_t end() const { return {}; }

 private:
  void destroy() {
    if (handle_) {
      handle_.destroy();
      handle_ = nullptr;
    }
  }

  handle_type handle_{};
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
    void return_value(T result) { value = std::move(result); }
    void unhandled_exception() noexcept { exception = std::current_exception(); }

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

  bool await_ready() const noexcept { return false; }
  std::coroutine_handle<> await_suspend(std::coroutine_handle<> caller) noexcept {
    handle_.promise().continuation = caller;
    return handle_;
  }
  T await_resume() {
    if (handle_.promise().exception) {
      std::rethrow_exception(handle_.promise().exception);
    }
    return *handle_.promise().value;
  }

  T take() {
    if (handle_.promise().exception) {
      std::rethrow_exception(handle_.promise().exception);
    }
    return *handle_.promise().value;
  }

  handle_type handle() const { return handle_; }

 private:
  void destroy() {
    if (handle_) {
      handle_.destroy();
      handle_ = nullptr;
    }
  }

  handle_type handle_{};
};

template <typename T>
T sync_wait(Task<T> task) {
  if (task.handle() && !task.handle().done()) {
    task.handle().resume();
  }
  return task.take();
}

Generator<int> countdown(int from) {
  for (int value = from; value > 0; --value) {
    co_yield value;
  }
}

Task<int> add(int left, int right) {
  co_return left + right;
}

Task<int> compute() {
  const int first = co_await add(19, 2);
  co_return co_await add(first, 21);
}

int main() {
  for (int value : countdown(3)) {
    std::cout << value << ' ';
  }
  std::cout << '\n';
  std::cout << sync_wait(compute()) << '\n';
}

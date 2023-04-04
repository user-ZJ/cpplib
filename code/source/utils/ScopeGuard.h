#ifndef BASE_ScopeGuard_UTIL_H_
#define BASE_ScopeGuard_UTIL_H_

#include <cstdlib>
#include <stack>

namespace BASE_NAMESPACE {

template <typename F>
class ScopeGuard {
 public:
  explicit ScopeGuard(F &&f) : m_func(std::move(f)), m_dismiss(false) {}

  explicit ScopeGuard(const F &f) : m_func(f), m_dismiss(false) {}

  ~ScopeGuard() {
    if (!m_dismiss) m_func();
  }

  ScopeGuard(ScopeGuard &&rhs) : m_func(std::move(rhs.m_func)), m_dismiss(rhs.m_dismiss) {
    rhs.dismiss();
  }

  ScopeGuard() = delete;
  ScopeGuard(const ScopeGuard &) = delete;
  ScopeGuard &operator=(const ScopeGuard &) = delete;
  void dismiss() {
    m_dismiss = true;
  }

 private:
  F m_func;
  bool m_dismiss;
};
}  // namespace BASE_NAMESPACE

#endif
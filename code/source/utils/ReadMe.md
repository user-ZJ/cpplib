### 时间统计

```cpp
#include<chrono>
auto begin_t = std::chrono::steady_clock::now();
auto finish_t = std::chrono::steady_clock::now();
//统计耗时，单位为ms
double timecost = std::chrono::duration<double,std::milli>(finish_t-begin_t).count();
std::chrono::duration_cast<std::chrono::milliseconds>(time_now-time_start_).count();
//统计耗时，单位为s
std::chrono::duration<double>{stop - start}.count()
```


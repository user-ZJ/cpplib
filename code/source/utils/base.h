/*
 * @Author: zack 
 * @Date: 2021-10-05 10:27:24 
 * @Last Modified by: zack
 * @Last Modified time: 2021-10-05 10:27:46
 */
#ifndef _BASE_H_
#define _BASE_H_
#include <thread>

namespace BASE_NAMESPACE {

using int8 = int8_t;
using int16 = int16_t;
using int32 = int32_t;
using int64 = int64_t;

using uint8 = uint8_t;
using uint16 = uint16_t;
using uint32 = uint32_t;
using uint64 = uint64_t;

typedef float   float32;
typedef double double64;

#define DISALLOW_COPY_AND_ASSIGN(Type) \
    Type(const Type &) = delete;         \
    Type &operator=(const Type &) = delete;

inline int get_num_physical_cores() {
    unsigned int n_threads = std::thread::hardware_concurrency();
    return n_threads > 0 ? (n_threads <= 4 ? n_threads : n_threads / 2) : 4;
}

}
#endif
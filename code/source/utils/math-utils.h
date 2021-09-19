#pragma once
#include <cmath>

inline double Log(double x) { return log(x); }                                                                                                                                                        
inline float Log(float x) { return logf(x); }


/// return abs(a - b) <= relative_tolerance * (abs(a)+abs(b)).
static inline bool ApproxEqual(float a, float b,
                               float relative_tolerance = 0.001) {
  // a==b handles infinities.
  if (a == b)
    return true;
  float diff = std::abs(a - b);
  if (diff == std::numeric_limits<float>::infinity() || diff != diff)
    return false; // diff is +inf or nan.
  return (diff <= relative_tolerance * (std::abs(a) + std::abs(b)));
}

// State for thread-safe random number generator
struct RandomState {
  RandomState();
  unsigned seed;
};

int Rand(struct RandomState *state) {
#if !defined(_POSIX_THREAD_SAFE_FUNCTIONS)
  // On Windows and Cygwin, just call Rand()
  return rand();
#else
  if (state) {
    return rand_r(&(state->seed));
  } else {
    std::lock_guard<std::mutex> lock(_RandMutex);
    return rand();
  }
#endif
}

// 产生min_val到max_val之间的一个随机整数
inline int RandInt(int min_val, int max_val, struct RandomState *state = NULL) {
  if (max_val <= min_val)
    return min_val;

#ifdef _MSC_VER
  // RAND_MAX is quite small on Windows -> may need to handle larger numbers.
  if (RAND_MAX > (max_val - min_val) * 8) {
    // *8 to avoid large inaccuracies in probability, from the modulus...
    return min_val +
           ((unsigned int)Rand(state) % (unsigned int)(max_val + 1 - min_val));
  } else {
    if ((unsigned int)(RAND_MAX * RAND_MAX) >
        (unsigned int)((max_val + 1 - min_val) * 8)) {
      // *8 to avoid inaccuracies in probability, from the modulus...
      return min_val + ((unsigned int)((Rand(state) + RAND_MAX * Rand(state))) %
                        (unsigned int)(max_val + 1 - min_val));
    } else {
      KALDI_ERR << "rand_int failed because we do not support such large "
                   "random numbers. (Extend this function).";
    }
  }
#else
  return min_val + (static_cast<int32>(Rand(state)) %
                    static_cast<int32>(max_val + 1 - min_val));
#endif
}
// Returns true with probability "prob",
inline bool WithProb(float prob, struct RandomState *state = NULL) {
  if (!(prob >= 0 &&
        prob <=
            1.1)) { // prob should be <= 1.0,but we allow slightly larger values
                    // that could arise from roundoff in previous calculations.
    return false;
  }
  if (!(RAND_MAX > 128 * 128))
    return false;
  if (prob == 0)
    return false;
  else if (prob == 1.0)
    return true;
  else if (prob * RAND_MAX < 128.0) {
    // prob is very small but nonzero, and the "main algorithm"
    // wouldn't work that well.  So: with probability 1/128, we
    // return WithProb (prob * 128), else return false.
    if (Rand(state) < RAND_MAX / 128) { // with probability 128...
      // Note: we know that prob * 128.0 < 1.0, because
      // we asserted RAND_MAX > 128 * 128.
      return WithProb(prob * 128.0);
    } else {
      return false;
    }
  } else {
    return (Rand(state) < ((RAND_MAX + static_cast<float>(1.0)) * prob));
  }
}

/// 随机0-1之间的浮点数
inline float RandUniform(struct RandomState *state = NULL) {
  return static_cast<float>((Rand(state) + 1.0) / (RAND_MAX + 2.0));
}

inline float RandGauss(struct RandomState *state = NULL) {
  return static_cast<float>(sqrtf(-2 * Log(RandUniform(state))) *
                            cosf(2 * M_PI * RandUniform(state)));
}




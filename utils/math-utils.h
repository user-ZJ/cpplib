#ifndef MATH_UTILS_H_
#define MATH_UTILS_H_

/// return abs(a - b) <= relative_tolerance * (abs(a)+abs(b)).
static inline bool ApproxEqual(float a, float b,
                               float relative_tolerance = 0.001) {
  // a==b handles infinities.
  if (a == b) return true;
  float diff = std::abs(a-b);
  if (diff == std::numeric_limits<float>::infinity()
      || diff != diff) return false;  // diff is +inf or nan.
  return (diff <= relative_tolerance*(std::abs(a)+std::abs(b)));
}

#endif

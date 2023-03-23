// Copyright (c) 2016 Network
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef BASE_FFT_H_
#define BASE_FFT_H_

namespace BASE_NAMESPACE {

// Fast Fourier Transform

void make_sintbl(int n, float *sintbl);

void make_bitrev(int n, int *bitrev);

int fft(const int *bitrev, const float *sintbl, float *x, float *y, int n);

}  // namespace BASE_NAMESPACE

#endif  // FRONTEND_FFT_H_

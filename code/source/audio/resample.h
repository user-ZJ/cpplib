// feat/resample.h

// Copyright     2013  Pegah Ghahremani
//               2014  IMSL, PKU-HKUST (author: Wei Shi)
//               2014  Yanqing Sun, Junjie Wang
//               2014  Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cassert>
#include <cstdlib>
#include <string>
#include <vector>

namespace BASE_NAMESPACE {
/// @addtogroup  feat FeatureExtraction
/// @{

/**
   \file[resample.h]

   This header contains declarations of classes for resampling signals.  The
   normal cases of resampling a signal are upsampling and downsampling
   (increasing and decreasing the sample rate of a signal, respectively),
   although the ArbitraryResample class allows a more generic case where
   we want to get samples of a signal at uneven intervals (for instance,
   log-spaced).

   The input signal is always evenly spaced, say sampled with frequency S, and
   we assume the original signal was band-limited to S/2 or lower.  The n'th
   input sample x_n (with n = 0, 1, ...) is interpreted as the original
   signal's value at time n/S.

   For resampling, it is convenient to view the input signal as a
   continuous function x(t) of t, where each sample x_n becomes a delta function
   with magnitude x_n/S, at time n/S.  If we band limit this to the Nyquist
   frequency S/2, we can show that this is the same as the original signal
   that was sampled. [assuming the original signal was periodic and band
   limited.]  In general we want to bandlimit to lower than S/2, because
   we don't have a perfect filter and also because if we want to resample
   at a lower frequency than S, we need to bandlimit to below half of that.
   Anyway, suppose we want to bandlimit to C, with 0 < C < S/2.  The perfect
   rectangular filter with cutoff C is the sinc function,
   \f[         f(t) = 2C sinc(2Ct),                   \f]
   where sinc is the normalized sinc function \f$ sinc(t) = sin(pi t) / (pi t) \f$, with
  \f$  sinc(0) = 1 \f$.  This is not a practical filter, though, because it has
   infinite support.  At the cost of less-than-perfect rolloff, we can choose
   a suitable windowing function g(t), and use f(t) g(t) as the filter.  For
   a windowing function we choose raised-cosine (Hanning) window with support
   on [-w/2C, w/2C], where w >= 2 is an integer chosen by the user.  w = 1
   means we window the sinc function out to its first zero on the left and right,
   w = 2 means the second zero, and so on; we normally choose w to be at least two.
   We call this num_zeros, not w, in the code.

   Convolving the signal x(t) with this windowed filter h(t) = f(t)g(t) and evaluating the resulting
   signal s(t) at an arbitrary time t is easy: we have
    \f[          s(t) = 1/S \sum_n x_n h(t - n/S)        \f].
   (note: the sign of t - n/S might be wrong, but it doesn't matter as the filter
   and window are symmetric).
   This is true for arbitrary values of t.  What the class ArbitraryResample does
   is to allow you to evaluate the signal for specified values of t.
*/


class LinearResample {
 public:
  /// Constructor.  We make the input and output sample rates integers, because
  /// we are going to need to find a common divisor.  This should just remind
  /// you that they need to be integers.  The filter cutoff needs to be less
  /// than samp_rate_in_hz/2 and less than samp_rate_out_hz/2.  num_zeros
  /// controls the sharpness of the filter, more == sharper but less efficient.
  /// We suggest around 4 to 10 for normal use.
  LinearResample(int samp_rate_in_hz, int samp_rate_out_hz, float filter_cutoff_hz, int num_zeros);

  /// This function does the resampling.  If you call it with flush == true and
  /// you have never called it with flush == false, it just resamples the input
  /// signal (it resizes the output to a suitable number of samples).
  ///
  /// You can also use this function to process a signal a piece at a time.
  /// suppose you break it into piece1, piece2, ... pieceN.  You can call
  /// \code{.cc}
  /// Resample(piece1, &output1, false);
  /// Resample(piece2, &output2, false);
  /// Resample(piece3, &output3, true);
  /// \endcode
  /// If you call it with flush == false, it won't output the last few samples
  /// but will remember them, so that if you later give it a second piece of
  /// the input signal it can process it correctly.
  /// If your most recent call to the object was with flush == false, it will
  /// have internal state; you can remove this by calling Reset().
  /// Empty input is acceptable.
  void Resample(const std::vector<float> &input, bool flush, std::vector<float> *output);

  /// Calling the function Reset() resets the state of the object prior to
  /// processing a new signal; it is only necessary if you have called
  /// Resample(x, y, false) for some signal, leading to a remainder of the
  /// signal being called, but then abandon processing the signal before calling
  /// Resample(x, y, true) for the last piece.  Call it unnecessarily between
  /// signals will not do any harm.
  void Reset();

  //// Return the input and output sampling rates (for checks, for example)
  inline int GetInputSamplingRate() {
    return samp_rate_in_;
  }
  inline int GetOutputSamplingRate() {
    return samp_rate_out_;
  }

 private:
  /// This function outputs the number of output samples we will output
  /// for a signal with "input_num_samp" input samples.  If flush == true,
  /// we return the largest n such that
  /// (n/samp_rate_out_) is in the interval [ 0, input_num_samp/samp_rate_in_ ),
  /// and note that the interval is half-open.  If flush == false,
  /// define window_width as num_zeros / (2.0 * filter_cutoff_);
  /// we return the largest n such that (n/samp_rate_out_) is in the interval
  /// [ 0, input_num_samp/samp_rate_in_ - window_width ).
  int64_t GetNumOutputSamples(int64_t input_num_samp, bool flush) const;

  /// Given an output-sample index, this function outputs to *first_samp_in the
  /// first input-sample index that we have a weight on (may be negative),
  /// and to *samp_out_wrapped the index into weights_ where we can get the
  /// corresponding weights on the input.
  inline void GetIndexes(int64_t samp_out, int64_t *first_samp_in, int *samp_out_wrapped) const;

  void SetRemainder(const std::vector<float> &input);

  void SetIndexesAndWeights();

  float FilterFunc(float) const;

  // The following variables are provided by the user.
  int samp_rate_in_;
  int samp_rate_out_;
  float filter_cutoff_;
  int num_zeros_;

  int input_samples_in_unit_;   ///< The number of input samples in the
                                ///< smallest repeating unit: num_samp_in_ =
                                ///< samp_rate_in_hz / Gcd(samp_rate_in_hz,
                                ///< samp_rate_out_hz)
  int output_samples_in_unit_;  ///< The number of output samples in the
                                ///< smallest repeating unit: num_samp_out_ =
                                ///< samp_rate_out_hz / Gcd(samp_rate_in_hz,
                                ///< samp_rate_out_hz)

  /// The first input-sample index that we sum over, for this output-sample
  /// index.  May be negative; any truncation at the beginning is handled
  /// separately.  This is just for the first few output samples, but we can
  /// extrapolate the correct input-sample index for arbitrary output samples.
  std::vector<int> first_index_;

  /// Weights on the input samples, for this output-sample index.
  std::vector<std::vector<float>> weights_;

  // the following variables keep track of where we are in a particular signal,
  // if it is being provided over multiple calls to Resample().

  int64_t input_sample_offset_;         ///< The number of input samples we have
                                        ///< already received for this signal
                                        ///< (including anything in remainder_)
  int64_t output_sample_offset_;        ///< The number of samples we have already
                                        ///< output for this signal.
  std::vector<float> input_remainder_;  ///< A small trailing part of the
                                        ///< previously seen input signal.
};

/**
   Downsample or upsample a waveform. This is a convenience wrapper for the
   class 'LinearResample'.
   The low-pass filter cutoff used in 'LinearResample' is 0.99 of the Nyquist,
   where the Nyquist is half of the minimum of (orig_freq, new_freq).  The
   resampling is done with a symmetric FIR filter with N_z (number of zeros)
   as 6.

   We compared the downsampling results with those from the sox resampling
   toolkit.
   Sox's design is inspired by Laurent De Soras' paper,
   https://ccrma.stanford.edu/~jos/resample/Implementation.html

   Note: we expect that while orig_freq and new_freq are of type BaseFloat, they
   are actually required to have exact integer values (like 16000 or 8000) with
   a ratio between them that can be expressed as a rational number with
   reasonably small integer factors.
*/
void ResampleWaveform(int orig_freq, const std::vector<float> &wave, int new_freq, std::vector<float> *new_wave);

/// @} End of "addtogroup feat"
}  // namespace BASE_NAMESPACE

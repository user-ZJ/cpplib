// featbin/compute-kaldi-pitch-feats.cc

// Copyright 2013        Pegah Ghahremani
//           2013-2014   Johns Hopkins University (author: Daniel Povey)
//
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

#include "base/kaldi-common.h"
#include "feat/pitch-functions.h"
#include "feat/wave-reader.h"
#include "util/common-utils.h"

namespace kaldi{

int compute_kaldi_pitch_feats(const WaveData &wave_data,
                              const PitchExtractionOptions &pitch_opts,
                              const std::string &id,
                              Matrix<BaseFloat> &features) {
  try {

    int32 num_done = 0, num_err = 0;

    std::string utt = id;
    // const WaveData &wave_data = wav_reader.Value();

    int32 num_chan = wave_data.Data().NumRows(), this_chan = 0;

    if (pitch_opts.samp_freq != wave_data.SampFreq()) {
      KALDI_ERR << "Sample frequency mismatch: you specified "
                << pitch_opts.samp_freq << " but data has "
                << wave_data.SampFreq() << " (use --sample-frequency "
                << "option).  Utterance is " << utt;
      return -1;
    }

    SubVector<BaseFloat> waveform(wave_data.Data(), this_chan);
    try {
      ComputeKaldiPitch(pitch_opts, waveform, &features);
    } catch (...) {
      KALDI_WARN << "Failed to compute pitch for utterance " << utt;
      num_err++;
      return -1;
    }
    num_done++;

    KALDI_LOG << "Done " << num_done << " utterances, " << num_err
              << " with errors.";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

}
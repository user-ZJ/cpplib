// featbin/compute-mfcc-feats.cc

// Copyright 2009-2012  Microsoft Corporation
//                      Johns Hopkins University (author: Daniel Povey)

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
#include "feat/feature-mfcc.h"
#include "feat/wave-reader.h"
#include "util/common-utils.h"
#include "kaldi-featlibs.h"

namespace kaldi{

int compute_mfcc_feats(const WaveData &wave_data,const MfccOptions &mfcc_opts,const std::string &id,Matrix<float> &features) {
  try {

    // Define defaults for global options.
    // int32 srand_seed = 0;
    // srand(srand_seed);
    // mfcc_opts.frame_opts.dither=0.0;
    bool subtract_mean = false;
    BaseFloat vtln_warp = 1.0;
    int32 channel = -1;
    BaseFloat min_duration = 0.0;

    Mfcc mfcc(mfcc_opts);


    int32 num_utts = 0, num_success = 0;
    {
      num_utts++;
      std::string utt = id;
      if (wave_data.Duration() < min_duration) {
        KALDI_WARN << "File: " << utt << " is too short ("
                   << wave_data.Duration() << " sec): producing no output.";
        return -1;
      }
      int32 num_chan = wave_data.Data().NumRows(), this_chan = channel;
      {  // This block works out the channel (0=left, 1=right...)
        KALDI_ASSERT(num_chan > 0);  // should have been caught in
        // reading code if no channels.
        if (channel == -1) {
          this_chan = 0;
          if (num_chan != 1)
            KALDI_WARN << "Channel not specified but you have data with "
                       << num_chan  << " channels; defaulting to zero";
        } else {
          if (this_chan >= num_chan) {
            KALDI_WARN << "File with id " << utt << " has "
                       << num_chan << " channels but you specified channel "
                       << channel << ", producing no output.";
            return -1;
          }
        }
      }
      BaseFloat vtln_warp_local = vtln_warp;

      SubVector<BaseFloat> waveform(wave_data.Data(), this_chan);
      try {
        mfcc.ComputeFeatures(waveform, wave_data.SampFreq(),
                             vtln_warp_local, &features);
      } catch (...) {
        KALDI_WARN << "Failed to compute features for utterance " << utt;
        return -1;
      }
      num_success++;
    }
    return (num_success != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

};
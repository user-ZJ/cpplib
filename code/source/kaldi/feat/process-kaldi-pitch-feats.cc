// featbin/process-kaldi-pitch-feats.cc

// Copyright 2013   Pegah Ghahremani
//                  Johns Hopkins University (author: Daniel Povey)
//           2014   IMSL, PKU-HKUST (author: Wei Shi)
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

int process_kaldi_pitch_feats(const Matrix<BaseFloat> &features,
                              const ProcessPitchOptions &process_opts,
                              const std::string &id,
                              Matrix<BaseFloat> &processed_feats) {
  try {

    int32 srand_seed = 0;
    srand(srand_seed);

    int32 num_done = 0;

    std::string utt = id;
    processed_feats.Resize(features.NumRows(), features.NumCols());
    processed_feats.CopyFromMat(features);
    ProcessPitch(process_opts, features, &processed_feats);
    num_done++;

    KALDI_LOG << "Post-processed pitch for " << num_done << " utterances.";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

}

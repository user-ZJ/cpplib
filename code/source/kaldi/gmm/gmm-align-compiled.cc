// gmmbin/gmm-align-compiled.cc

// Copyright 2009-2013  Microsoft Corporation
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
#include "decoder/decoder-wrappers.h"
#include "fstext/fstext-lib.h"
#include "gmm/am-diag-gmm.h"
#include "gmm/decodable-am-diag-gmm.h"
#include "hmm/hmm-utils.h"
#include "hmm/transition-model.h"
#include "kaldi-gmmlibs.h"
#include "lat/kaldi-lattice.h" // for {Compact}LatticeArc
#include "util/common-utils.h"

namespace kaldi {

int gmm_align_compiled(const TransitionModel &trans_model,
                       const AmDiagGmm &am_gmm,
                       fst::VectorFst<fst::StdArc> decode_fst,
                       const Matrix<float> &features, const std::string &id,
                       std::vector<int32> *alignment) {
  try {
    // using fst::SymbolTable;
    // using fst::VectorFst;
    // using fst::StdArc;

    // ParseOptions po(usage);
    AlignConfig align_config;
    align_config.beam = 10;
    align_config.retry_beam = 40;
    align_config.careful = false;
    BaseFloat acoustic_scale = 0.1;
    BaseFloat transition_scale = 1.0;
    BaseFloat self_loop_scale = 0.1;

    // TransitionModel trans_model;
    // AmDiagGmm am_gmm;
    // {
    //   bool binary;
    //   Input ki(model_in_filename, &binary);
    //   trans_model.Read(ki.Stream(), binary);
    //   am_gmm.Read(ki.Stream(), binary);
    // }

    int num_done = 0, num_err = 0, num_retry = 0;
    double tot_like = 0.0;
    kaldi::int64 frame_count = 0;

    std::string utt = id;

    if (features.NumRows() == 0) {
      KALDI_WARN << "Zero-length utterance: " << utt;
      num_err++;
      return -1;
    }

    {                                   // Add transition-probs to the FST.
      std::vector<int32> disambig_syms; // empty.
      AddTransitionProbs(trans_model, disambig_syms, transition_scale,
                         self_loop_scale, &decode_fst);
    }

    DecodableAmDiagGmmScaled gmm_decodable(am_gmm, trans_model, features,
                                           acoustic_scale);

    KALDI_LOG << utt;
    // std::vector<int32> alignment;
    alignment->clear();
    AlignUtteranceWrapper1(align_config, utt, acoustic_scale, &decode_fst,
                          &gmm_decodable, alignment, &num_done, &num_err,
                          &num_retry, &tot_like, &frame_count);

    KALDI_LOG << "Overall log-likelihood per frame is "
              << (tot_like / frame_count) << " over " << frame_count
              << " frames.";
    KALDI_LOG << "Retried " << num_retry << " out of " << (num_done + num_err)
              << " utterances.";
    KALDI_LOG << "Done " << num_done << ", errors on " << num_err;
    return (num_done != 0 ? 0 : 1);
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

}; // namespace kaldi

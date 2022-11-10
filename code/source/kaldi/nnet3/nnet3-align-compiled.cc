// nnet2bin/nnet-align-compiled.cc

// Copyright 2009-2012     Microsoft Corporation
//                         Johns Hopkins University (author: Daniel Povey)
//                2015     Vijayaditya Peddinti
//                2015-16  Vimal Manohar

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
#include "decoder/training-graph-compiler.h"
#include "fstext/fstext-lib.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/hmm-utils.h"
#include "hmm/transition-model.h"
#include "lat/kaldi-lattice.h"
#include "nnet3/kaldi-nnet3libs.h"
#include "nnet3/nnet-am-decodable-simple.h"
#include "nnet3/nnet-utils.h"
#include "util/common-utils.h"

namespace kaldi {

int nnet3_align_compiled(const TransitionModel &trans_model,
                         AmNnetSimple &am_nnet,
                         fst::VectorFst<fst::StdArc> decode_fst,
                         const Matrix<BaseFloat> &features,
                         const Matrix<BaseFloat> &online_ivectors,
                         int32 online_ivector_period, AlignConfig &align_config,
                         NnetSimpleComputationOptions &decodable_opts,
                         const std::string &id, std::vector<int32> &alignment) {
  try {

    using namespace nnet3;
    typedef kaldi::int32 int32;
    using fst::StdArc;
    using fst::SymbolTable;
    using fst::VectorFst;

    BaseFloat transition_scale = 1.0;
    BaseFloat self_loop_scale = 0.1;
    std::string per_frame_acwt_wspecifier;

    std::string ivector_rspecifier, online_ivector_rspecifier,
        utt2spk_rspecifier;

    int num_done = 0, num_err = 0, num_retry = 0;
    double tot_like = 0.0;
    kaldi::int64 frame_count = 0;

    {
      // TransitionModel trans_model;
      // AmNnetSimple am_nnet;
      // {
      //   bool binary;
      //   Input ki(model_in_filename, &binary);
      //   trans_model.Read(ki.Stream(), binary);
      //   am_nnet.Read(ki.Stream(), binary);
      // }
      SetBatchnormTestMode(true, &(am_nnet.GetNnet()));
      SetDropoutTestMode(true, &(am_nnet.GetNnet()));
      CollapseModel(CollapseModelConfig(), &(am_nnet.GetNnet()));
      // this compiler object allows caching of computations across
      // different utterances.
      CachingOptimizingCompiler compiler(am_nnet.GetNnet(),
                                         decodable_opts.optimize_config);
      {
        std::string utt = id;

        if (features.NumRows() == 0) {
          KALDI_WARN << "Zero-length utterance: " << utt;
          num_err++;
          return -1;
        }

        // const Matrix<BaseFloat> *online_ivectors = NULL;
        const Vector<BaseFloat> *ivector = NULL;

        {                                   // Add transition-probs to the FST.
          std::vector<int32> disambig_syms; // empty.
          AddTransitionProbs(trans_model, disambig_syms, transition_scale,
                             self_loop_scale, &decode_fst);
        }

        DecodableAmNnetSimple nnet_decodable(
            decodable_opts, trans_model, am_nnet, features, ivector,
            &online_ivectors, online_ivector_period, &compiler);
        alignment.clear();
        AlignUtteranceWrapper1(align_config, utt, decodable_opts.acoustic_scale,
                               &decode_fst, &nnet_decodable, &alignment,
                               &num_done, &num_err, &num_retry, &tot_like,
                               &frame_count);
        KALDI_LOG<<"aligment size:"<<alignment.size();
      }
      KALDI_LOG << "Overall log-likelihood per frame is "
                << (tot_like / frame_count) << " over " << frame_count
                << " frames.";
      KALDI_LOG << "Retried " << num_retry << " out of " << (num_done + num_err)
                << " utterances.";
      KALDI_LOG << "Done " << num_done << ", errors on " << num_err;
    }

    return (num_done != 0 ? 0 : 1);
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

} // namespace kaldi

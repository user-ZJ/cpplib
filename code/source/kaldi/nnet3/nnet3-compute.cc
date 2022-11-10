// nnet3bin/nnet3-compute.cc

// Copyright 2012-2015   Johns Hopkins University (author: Daniel Povey)
//                2015   Vimal Manohar

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
#include "base/timer.h"
#include "nnet3/nnet-am-decodable-simple.h"
#include "nnet3/nnet-utils.h"
#include "util/common-utils.h"
#include "nnet3/kaldi-nnet3libs.h"

namespace kaldi {

using namespace nnet3;

int nnet3_compute(const std::string &nnet_rxfilename,
                  const Matrix<BaseFloat> &features,
                  const Matrix<BaseFloat> online_ivectors,
                  int online_ivector_period,
                  NnetSimpleComputationOptions &opts, const std::string &id,
                  Matrix<BaseFloat> &matrix) {
  try {
    
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    // NnetSimpleComputationOptions opts;
    opts.acoustic_scale = 1.0; // by default do no scaling.

    bool apply_exp = false, use_priors = false;
    std::string use_gpu = "no";


    // int32 online_ivector_period = 10;


    Nnet raw_nnet;
    AmNnetSimple am_nnet;
    if (use_priors) {
      bool binary;
      TransitionModel trans_model;
      Input ki(nnet_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_nnet.Read(ki.Stream(), binary);
    } else {
      ReadKaldiObject(nnet_rxfilename, &raw_nnet);
    }
    Nnet &nnet = (use_priors ? am_nnet.GetNnet() : raw_nnet);
    SetBatchnormTestMode(true, &nnet);
    SetDropoutTestMode(true, &nnet);
    CollapseModel(CollapseModelConfig(), &nnet);

    Vector<BaseFloat> priors;
    if (use_priors)
      priors = am_nnet.Priors();


    CachingOptimizingCompiler compiler(nnet, opts.optimize_config);

    int32 num_success = 0, num_fail = 0;
    int64 frame_count = 0;



    {
      std::string utt = id;
      if (features.NumRows() == 0) {
        KALDI_WARN << "Zero-length utterance: " << utt;
        num_fail++;
        return -1;
      }

      const Vector<BaseFloat> *ivector = NULL;


      DecodableNnetSimple nnet_computer(opts, nnet, priors, features, &compiler,
                                        ivector, &online_ivectors,
                                        online_ivector_period);

      matrix.Resize(nnet_computer.NumFrames(), nnet_computer.OutputDim());
      for (int32 t = 0; t < nnet_computer.NumFrames(); t++) {
        SubVector<BaseFloat> row(matrix, t);
        nnet_computer.GetOutputForFrame(t, &row);
      }

      if (apply_exp)
        matrix.ApplyExp();

      // matrix_writer.Write(utt, matrix);

      frame_count += features.NumRows();
      num_success++;
    }

    KALDI_LOG << "Done " << num_success << " utterances, failed for "
              << num_fail;

    if (num_success != 0)
      return 0;
    else
      return 1;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
} // namespace kaldi

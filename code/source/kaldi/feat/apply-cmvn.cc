// featbin/apply-cmvn.cc

// Copyright 2009-2011  Microsoft Corporation
//                2014  Johns Hopkins University

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
#include "matrix/kaldi-matrix.h"
#include "transform/cmvn.h"
#include "util/common-utils.h"

namespace kaldi {

int apply_cmvn(const Matrix<float> &feat, const Matrix<double> &cmvn_stats,
               bool norm_means, bool norm_vars, const std::string &id,
               Matrix<float> &norm_feat) {
  try {

    bool reverse = false;
    std::string skip_dims_str;

    if (norm_vars && !norm_means) {
      KALDI_ERR << "You cannot normalize the variance but not the mean.";
      return -1;
    }

    if (!norm_means) {
      // CMVN is a no-op, we're not doing anything.  Just echo the input
      // don't even uncompress, if it was a CompressedMatrix.

      int32 num_done = 0;
      norm_feat = feat;
      num_done++;
      KALDI_LOG << "Copied " << num_done << " utterances.";
      return 0;
    }

    int32 num_done = 0, num_err = 0;

    std::string utt = id;
    norm_feat = feat;
    if (norm_means) {
      if (reverse) {
        ApplyCmvnReverse(cmvn_stats, norm_vars, &norm_feat);
      } else {
        ApplyCmvn(cmvn_stats, norm_vars, &norm_feat);
      }
    }
    num_done++;

    if (norm_vars)
      KALDI_LOG << "Applied cepstral mean and variance normalization to "
                << num_done << " utterances, errors on " << num_err;
    else
      KALDI_LOG << "Applied cepstral mean normalization to " << num_done
                << " utterances, errors on " << num_err;
    return (num_done != 0 ? 0 : 1);
  } catch (const std::exception &e) {
    KALDI_ERR << e.what();
    return -1;
  }
}

} // namespace kaldi

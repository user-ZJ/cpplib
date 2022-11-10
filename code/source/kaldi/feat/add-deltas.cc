// featbin/add-deltas.cc

// Copyright 2009-2011  Microsoft Corporation

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
#include "feat/feature-functions.h"
#include "kaldi-featlibs.h"
#include "matrix/kaldi-matrix.h"
#include "util/common-utils.h"

namespace kaldi {

int add_deltas(const Matrix<float> &feats, const std::string &id,
         Matrix<float> &new_feats) {
  try {
    DeltaFeaturesOptions opts;
    int32 truncate = 0;

    std::string key = id;

    if (feats.NumRows() == 0) {
      KALDI_WARN << "Empty feature matrix for key " << key;
      return -1;
    }
    if (truncate != 0) {
      if (truncate > feats.NumCols())
        KALDI_ERR << "Cannot truncate features as dimension " << feats.NumCols()
                  << " is smaller than truncation dimension.";
      SubMatrix<BaseFloat> feats_sub(feats, 0, feats.NumRows(), 0, truncate);
      ComputeDeltas(opts, feats_sub, &new_feats);
    } else
      ComputeDeltas(opts, feats, &new_feats);

    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

}; // namespace kaldi

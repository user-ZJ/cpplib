// featbin/compute-cmvn-stats.cc

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
#include "matrix/kaldi-matrix.h"
#include "transform/cmvn.h"
#include "util/common-utils.h"
#include "kaldi-featlibs.h"

namespace kaldi {

bool AccCmvnStatsWrapper(const std::string &utt,
                         const MatrixBase<BaseFloat> &feats,
                         RandomAccessBaseFloatVectorReader *weights_reader,
                         Matrix<double> *cmvn_stats) {
  if (!weights_reader->IsOpen()) {
    AccCmvnStats(feats, NULL, cmvn_stats);
    return true;
  } else {
    if (!weights_reader->HasKey(utt)) {
      KALDI_WARN << "No weights available for utterance " << utt;
      return false;
    }
    const Vector<BaseFloat> &weights = weights_reader->Value(utt);
    if (weights.Dim() != feats.NumRows()) {
      KALDI_WARN << "Weights for utterance " << utt << " have wrong dimension "
                 << weights.Dim() << " vs. " << feats.NumRows();
      return false;
    }
    AccCmvnStats(feats, &weights, cmvn_stats);
    return true;
  }
}

int compute_cmvn_stats(const Matrix<float> &feats, const std::string &id,Matrix<double> &stats) {
  try {

    std::string utt = id;
    // Matrix<double> double_stats;
    InitCmvnStats(feats.NumCols(), &stats);
    AccCmvnStats(feats, NULL, &stats);

    // stats->Resize(double_stats.NumRows(),double_stats.NumCols());
    // stats->CopyFromMat(double_stats);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

} // namespace kaldi

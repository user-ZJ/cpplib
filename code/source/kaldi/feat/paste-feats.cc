// featbin/paste-feats.cc

// Copyright 2012 Korbinian Riedhammer
//           2013 Brno University of Technology (Author: Karel Vesely)
//           2013 Johns Hopkins University (Author: Daniel Povey)

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
#include "util/common-utils.h"

namespace kaldi {

// returns true if successfully appended.
bool AppendFeats(const std::vector<Matrix<BaseFloat>> &in,
                 const std::string &utt, int32 tolerance,
                 Matrix<BaseFloat> *out) {
  // Check the lengths
  int32 min_len = in[0].NumRows(), max_len = in[0].NumRows(),
        tot_dim = in[0].NumCols();
  for (int32 i = 1; i < in.size(); i++) {
    int32 len = in[i].NumRows(), dim = in[i].NumCols();
    tot_dim += dim;
    if (len < min_len)
      min_len = len;
    if (len > max_len)
      max_len = len;
  }
  if (max_len - min_len > tolerance || min_len == 0) {
    KALDI_WARN << "Length mismatch " << max_len << " vs. " << min_len
               << (utt.empty() ? "" : " for utt ") << utt
               << " exceeds tolerance " << tolerance;
    out->Resize(0, 0);
    return false;
  }
  if (max_len - min_len > 0) {
    KALDI_VLOG(2) << "Length mismatch " << max_len << " vs. " << min_len
                  << (utt.empty() ? "" : " for utt ") << utt
                  << " within tolerance " << tolerance;
  }
  out->Resize(min_len, tot_dim);
  int32 dim_offset = 0;
  for (int32 i = 0; i < in.size(); i++) {
    int32 this_dim = in[i].NumCols();
    out->Range(0, min_len, dim_offset, this_dim)
        .CopyFromMat(in[i].Range(0, min_len, 0, this_dim));
    dim_offset += this_dim;
  }
  return true;
}

int paste_feats(const std::vector<Matrix<BaseFloat>> &feats,
                const std::string &id, Matrix<BaseFloat> &output) {
  try {

    int32 length_tolerance = 2;

    {
      if (!AppendFeats(feats, id, length_tolerance, &output))
        return -1; // it will have printed a warning.
      KALDI_LOG << "paste feat " << id;
      return 0;
    }
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

} // namespace kaldi





/*
  Testing:

cat <<EOF >1.mat
[ 0 1 2
  3 4 5
  8 9 10 ]
EOF
cat <<EOF > 2.mat
 [ 0 1
   2 3 ]
EOF
paste-feats --length-tolerance=1 --binary=false 1.mat 2.mat 3a.mat
cat <<EOF > 3b.mat
 [ 0 1 2 0 1
   3 4 5 2 3 ]
EOF
cmp <(../bin/copy-matrix 3b.mat -) <(../bin/copy-matrix 3a.mat -) || echo 'Bad!'

paste-feats --length-tolerance=1 'scp:echo foo 1.mat|' 'scp:echo foo 2.mat|'
'scp,t:echo foo 3a.mat|' cmp <(../bin/copy-matrix 3b.mat -) <(../bin/copy-matrix
3a.mat -) || echo 'Bad!'

rm {1,2,3?}.mat
 */

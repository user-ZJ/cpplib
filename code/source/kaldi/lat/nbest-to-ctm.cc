// latbin/nbest-to-ctm.cc

// Copyright 2012-2016  Johns Hopkins University (Author: Daniel Povey)

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
#include "kaldi-latlibs.h"
#include "lat/lattice-functions.h"
#include "util/common-utils.h"

namespace kaldi {

int nbest_to_ctm(const CompactLattice &clat, const std::string &id,
                 std::vector<std::tuple<float, float, int>> *ctm) {
  // cts format is {start_time,time_len,word_id}
  try {

    bool print_silence = false;
    BaseFloat frame_shift = 0.01;
    int32 precision = 2;

    if (frame_shift < 0.01 && precision <= 2)
      precision = 3;
    if (frame_shift < 0.001 && precision <= 3)
      precision = 4;

    ctm->clear();

    int32 n_done = 0, n_err = 0;

    std::string key = id;
    // CompactLattice clat = clat_reader.Value();

    std::vector<int32> words, times, lengths;

    if (!CompactLatticeToWordAlignment(clat, &words, &times, &lengths)) {
      n_err++;
      KALDI_WARN << "Format conversion failed for key " << key;
    } else {
      KALDI_ASSERT(words.size() == times.size() &&
                   words.size() == lengths.size());
      for (size_t i = 0; i < words.size(); i++) {
        if (words[i] == 0 &&
            !print_silence) // Don't output anything for <eps> links, which
          continue;         // correspond to silence....
        ctm->push_back(std::make_tuple(frame_shift * times[i],
                                       frame_shift * lengths[i], words[i]));
      }
      n_done++;
    }

    KALDI_LOG << "Converted " << n_done << " linear lattices to ctm format; "
              << n_err << " had errors.";
    return (n_done != 0 ? 0 : 1);
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

} // namespace kaldi

// bin/ali-to-phones.cc

// Copyright 2009-2011  Microsoft Corporation
//           2015       IMSL, PKU-HKUST (author: Wei Shi)

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
#include "fst/fstlib.h"
#include "hmm/hmm-utils.h"
#include "hmm/transition-model.h"
#include "util/common-utils.h"

namespace kaldi {

int ali_to_phones_pair(const TransitionModel &trans_model,
                       const std::vector<int32> &alignment,
                       const std::string &id,
                       std::vector<std::pair<int32, int32>> &pairs) {

  try {

    bool per_frame = false;
    bool write_lengths = true;
    bool ctm_output = false;
    BaseFloat frame_shift = 0.01;

    std::string empty;

    int32 n_done = 0;

    std::string key = id;

    std::vector<std::vector<int32>> split;
    SplitToPhones(trans_model, alignment, &split);

    std::vector<std::pair<int32, int32>> pairs;
    for (size_t i = 0; i < split.size(); i++) {
      if (split[i].empty()) {
        return -1;
      }
      int32 phone = trans_model.TransitionIdToPhone(split[i][0]);
      int32 num_repeats = split[i].size();
      pairs.push_back(std::make_pair(phone, num_repeats));
    }

    n_done++;
    KALDI_LOG << "Done " << n_done << " utterances.";
    return 0;
  } catch (const std::exception &e) {
    LOG(ERROR) << e.what();
    return -1;
  }
}

int ali_to_phones_ctm(const TransitionModel &trans_model,
                      const std::vector<int32> &alignment,
                      const std::string &id,
                      std::vector<std::tuple<float, float, int32>> &ctm) {
  // ctm是一个列表，列表中每个元素为 (start,duration,phone_id)
  try {

    BaseFloat frame_shift = 0.01;

    int32 n_done = 0;

    std::string key = id;

    std::vector<std::vector<int32>> split;
    SplitToPhones(trans_model, alignment, &split);

    ctm.clear();
    BaseFloat phone_start = 0.0;
    for (size_t i = 0; i < split.size(); i++) {
      if (split[i].empty()) {
        return -1;
      }
      int32 phone = trans_model.TransitionIdToPhone(split[i][0]);
      int32 num_repeats = split[i].size();
      ctm.push_back(
          std::make_tuple(phone_start, (frame_shift * num_repeats), phone));
      phone_start += frame_shift * num_repeats;
    }

    n_done++;

    KALDI_LOG << "Done " << n_done << " utterances.";
    return 0;
  } catch (const std::exception &e) {
    LOG(ERROR) << e.what();
    return -1;
  }
}

int ali_to_phones_frame(const TransitionModel &trans_model,
                        const std::vector<int32> &alignment,
                        const std::string &id, std::vector<int32> &phones) {

  try {

    bool per_frame = true;
    bool write_lengths = false;
    bool ctm_output = false;
    BaseFloat frame_shift = 0.01;

    int32 n_done = 0;

    std::string key = id;

    std::vector<std::vector<int32>> split;
    SplitToPhones(trans_model, alignment, &split);

    phones.clear();
    for (size_t i = 0; i < split.size(); i++) {
      if (split[i].empty()) {
        return -1;
      }
      int32 phone = trans_model.TransitionIdToPhone(split[i][0]);
      int32 num_repeats = split[i].size();
      if (per_frame)
        for (int32 j = 0; j < num_repeats; j++)
          phones.push_back(phone);
      else
        phones.push_back(phone);
    }

    n_done++;

    KALDI_LOG << "Done " << n_done << " utterances.";
    return 0;
  } catch (const std::exception &e) {
    LOG(ERROR) << e.what();
    return -1;
  }
}

} // namespace kaldi

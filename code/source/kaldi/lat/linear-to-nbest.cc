// latbin/linear-to-nbest.cc

// Copyright 2012  Johns Hopkins University (Author: Daniel Povey)

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
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"
#include "util/common-utils.h"
#include "kaldi-latlibs.h"

namespace kaldi {
void MakeLatticeFromLinear(const std::vector<int32> &ali,
                           const std::vector<int32> &words, BaseFloat lm_cost,
                           BaseFloat ac_cost, Lattice *lat_out) {
  typedef LatticeArc::StateId StateId;
  typedef LatticeArc::Weight Weight;
  typedef LatticeArc::Label Label;
  lat_out->DeleteStates();
  StateId cur_state = lat_out->AddState(); // will be 0.
  lat_out->SetStart(cur_state);
  for (size_t i = 0; i < ali.size() || i < words.size(); i++) {
    Label ilabel = (i < ali.size() ? ali[i] : 0);
    Label olabel = (i < words.size() ? words[i] : 0);
    StateId next_state = lat_out->AddState();
    lat_out->AddArc(cur_state,
                    LatticeArc(ilabel, olabel, Weight::One(), next_state));
    cur_state = next_state;
  }
  lat_out->SetFinal(cur_state, Weight(lm_cost, ac_cost));
}

int linear_to_nbest(const std::vector<int32> &ali,
                    const std::vector<int32> &words, const std::string &id,
                    CompactLattice *clat) {
  try {
    // using fst::SymbolTable;
    // using fst::VectorFst;
    // using fst::StdArc;

    int32 n_done = 0, n_err = 0;

    std::string key = id;

    BaseFloat ac_cost = 0.0, lm_cost = 0.0;
    Lattice lat;
    MakeLatticeFromLinear(ali, words, lm_cost, ac_cost, &lat);
    ConvertLattice(lat, clat);
    n_done++;
    KALDI_LOG << "Done " << n_done << " n-best entries ," << n_err
              << " had errors.";
    return (n_done != 0 ? 0 : 1);
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

} // namespace kaldi

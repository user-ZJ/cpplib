// latbin/lattice-align-words.cc

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
#include "kaldi-latlibs.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions-transition-model.h"
#include "lat/lattice-functions.h"
#include "lat/word-align-lattice.h"
#include "util/common-utils.h"

namespace kaldi {

int lattice_align_words(const std::vector<std::pair<int, std::string>> &word_boundary_int,
                        const TransitionModel &tmodel,
                        const CompactLattice &clat, const std::string &id,
                        CompactLattice *aligned_clat) {
  try {

    BaseFloat max_expand = 0.0;
    bool output_if_error = true;
    bool do_test = false;

    WordBoundaryInfoNewOpts opts;

    // TransitionModel tmodel;
    // ReadKaldiObject(model_rxfilename, &tmodel);

    // WordBoundaryInfo info(opts, word_boundary_rxfilename);
    WordBoundaryInfo info(opts);
    info.Init(word_boundary_int);

    int32 num_done = 0, num_err = 0;

    std::string key = id;

    int32 max_states;
    if (max_expand > 0)
      max_states = 1000 + max_expand * clat.NumStates();
    else
      max_states = 0;

    bool ok = WordAlignLattice(clat, tmodel, info, max_states, aligned_clat);

    if (do_test && ok)
      TestWordAlignedLattice(clat, tmodel, info, *aligned_clat);

    if (!ok) {
      num_err++;
      if (!output_if_error)
        KALDI_WARN << "Lattice for " << key
                   << " did not align correctly, producing no output.";
      else {
        if (aligned_clat->Start() != fst::kNoStateId) {
          KALDI_WARN << "Outputting partial lattice for " << key;
          TopSortCompactLatticeIfNeeded(aligned_clat);
        } else {
          KALDI_WARN << "Empty aligned lattice for " << key
                     << ", producing no output.";
        }
      }
    } else {
      if (aligned_clat->Start() == fst::kNoStateId) {
        num_err++;
        KALDI_WARN << "Lattice was empty for key " << key;
      } else {
        num_done++;
        KALDI_VLOG(2) << "Aligned lattice for " << key;
        TopSortCompactLatticeIfNeeded(aligned_clat);
      }
    }

    KALDI_LOG << "Successfully aligned " << num_done << " lattices; " << num_err
              << " had errors.";
    return (num_done > num_err? 0: 1); // We changed the error condition slightly here,
    // if there are errors in the word-boundary phones we can get situations
    // where most lattices give an error.
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

} // namespace kaldi

// latbin/lattice-to-phone-lattice.cc

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
#include "fstext/fstext-lib.h"
#include "hmm/transition-model.h"
#include "kaldi-latlibs.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"
#include "util/common-utils.h"

namespace kaldi {

int lattice_to_phone_lattice(const TransitionModel &tmodel,
                             const CompactLattice &aligned_clat,
                             std::string &id, CompactLattice *clat) {
  try {

    bool replace_words = true;

    int32 n_done = 0;


    // if (replace_words) {
    Lattice lat;
    ConvertLattice(aligned_clat, &lat);
    ConvertLatticeToPhones(tmodel,&lat); // this function replaces words -> phones
    ConvertLattice(lat, clat);
    // clat_writer.Write(clat_reader.Key(), clat);
    // } else { // replace transition-ids with phones.
    //   CompactLattice clat(clat_reader.Value());
    //   ConvertCompactLatticeToPhones(trans_model, &clat);
    //   // this function replaces transition-ids with phones.  We do it in the
    //   // CompactLattice form, in order to preserve the alignment of
    //   // transition-id sequences/phones-sequences to words [e.g. if you just
    //   // did lattice-align-words].
    //   clat_writer.Write(clat_reader.Key(), clat);
    // }
    n_done++;

    KALDI_LOG << "Done converting " << n_done << " lattices.";
    return (n_done != 0 ? 0 : 1);
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

} // namespace kaldi

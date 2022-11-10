#pragma once 
#include "base/kaldi-common.h"
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"
#include "util/common-utils.h"
#include "lat/lattice-functions-transition-model.h"


namespace kaldi{

int linear_to_nbest(const std::vector<int32> &ali,
                    const std::vector<int32> &words, 
                    const std::string &id,
                    CompactLattice *clat);

int lattice_align_words(const std::vector<std::pair<int, std::string>> &word_boundary_int,
                        const TransitionModel &tmodel,
                        const CompactLattice &clat, const std::string &id,
                        CompactLattice *aligned_clat);

int nbest_to_ctm(const CompactLattice &clat, const std::string &id,
                 std::vector<std::tuple<float, float, int>> *ctm);

int lattice_to_phone_lattice(const TransitionModel &tmodel,
                             const CompactLattice &aligned_clat,
                             std::string &id, CompactLattice *clat);

};
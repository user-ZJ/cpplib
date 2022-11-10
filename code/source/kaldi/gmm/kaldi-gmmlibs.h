#pragma once
#include "base/kaldi-common.h"
#include "decoder/decoder-wrappers.h"
#include "fstext/fstext-lib.h"
#include "gmm/am-diag-gmm.h"
#include "gmm/decodable-am-diag-gmm.h"
#include "hmm/hmm-utils.h"
#include "hmm/transition-model.h"
#include "kaldi-gmmlibs.h"
#include "lat/kaldi-lattice.h" // for {Compact}LatticeArc
#include "util/common-utils.h"

namespace kaldi{

int gmm_align_compiled(const TransitionModel &trans_model,
                       const AmDiagGmm &am_gmm,
                       fst::VectorFst<fst::StdArc> decode_fst,
                       const Matrix<float> &features, const std::string &id,
                       std::vector<int32> *alignment);

int gmm_boost_slience(TransitionModel *trans_model,AmDiagGmm *am_gmm);

};
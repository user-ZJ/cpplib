#ifndef KALDI_NNET3LIBS_H_
#define KALDI_NNET3LIBS_H_

#include "base/kaldi-common.h"
#include "nnet3/nnet-am-decodable-simple.h"
#include "decoder/decoder-wrappers.h"

namespace kaldi {
using namespace nnet3;

int nnet3_compute(const std::string &nnet_rxfilename,
                  const Matrix<BaseFloat> &features,
                  const Matrix<BaseFloat> online_ivectors,
                  int online_ivector_period,
                  NnetSimpleComputationOptions &opts, const std::string &id,
                  Matrix<BaseFloat> &matrix);

int nnet3_align_compiled(const TransitionModel &trans_model,
                         AmNnetSimple &am_nnet,
                         fst::VectorFst<fst::StdArc> decode_fst,
                         const Matrix<BaseFloat> &features,
                         const Matrix<BaseFloat> &online_ivectors,
                         int32 online_ivector_period, AlignConfig &align_config,
                         NnetSimpleComputationOptions &decodable_opts,
                         const std::string &id, std::vector<int32> &alignment);

}; // namespace kaldi
#endif

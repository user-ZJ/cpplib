#pragma once
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "decoder/training-graph-compiler.h"
#include "hmm/posterior.h"

namespace kaldi{

int compile_train_graph(const ContextDependency &ctx_dep,
                        const TransitionModel &trans_model,
                        fst::VectorFst<fst::StdArc> *lex_fst,
                        const std::vector<int32> &disambig_syms,
                        const std::vector<int32> &transcript,
                        const std::string &id,
                        fst::VectorFst<fst::StdArc> *decode_fst);

int ali_to_phones_pair(const TransitionModel &trans_model,
                       const std::vector<int32> &alignment,
                       const std::string &id,
                       std::vector<std::pair<int32, int32>> &pairs);
int ali_to_phones_ctm(const TransitionModel &trans_model,
                      const std::vector<int32> &alignment,
                      const std::string &id,
                      std::vector<std::tuple<float, float, int32>> &ctm);
int ali_to_phones_frame(const TransitionModel &trans_model,
                        const std::vector<int32> &alignment,
                        const std::string &id, std::vector<int32> &phones);

int compute_gop(const TransitionModel &trans_model, Matrix<BaseFloat> &probs,
                const std::vector<int32> &alignment,
                const std::vector<int32> &phone_map, const std::string &id,
                Posterior &posterior_gop);

};
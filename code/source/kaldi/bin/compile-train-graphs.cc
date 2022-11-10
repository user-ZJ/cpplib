// bin/compile-train-graphs.cc

// Copyright 2009-2012  Microsoft Corporation
//           2012-2015  Johns Hopkins University (Author: Daniel Povey)

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
#include "util/common-utils.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "decoder/training-graph-compiler.h"
#include "kaldi-binlibs.h"


namespace kaldi{

int compile_train_graph(const ContextDependency &ctx_dep,
                        const TransitionModel &trans_model,
                        fst::VectorFst<fst::StdArc> *lex_fst,
                        const std::vector<int32> &disambig_syms,
                        const std::vector<int32> &transcript,
                        const std::string &id,
                        fst::VectorFst<fst::StdArc> *decode_fst) {
  try {
    TrainingGraphCompilerOptions gopts;
    gopts.transition_scale = 0.0;  // Change the default to 0.0 since we will generally add the
    // transition probs in the alignment phase (since they change eacm time)
    gopts.self_loop_scale = 0.0;  // Ditto for self-loop probs.
    // std::string disambig_rxfilename;
    // gopts.Register(&po);

    // po.Register("batch-size", &batch_size,
    //             "Number of FSTs to compile at a time (more -> faster but uses "
    //             "more memory.  E.g. 500");
    // po.Register("read-disambig-syms", &disambig_rxfilename, "File containing "
    //             "list of disambiguation symbols in phone symbol table");
    
    // po.Read(argc, argv);

    // std::string tree_rxfilename = po.GetArg(1);
    // std::string model_rxfilename = po.GetArg(2);
    // std::string lex_rxfilename = po.GetArg(3);
    // std::string transcript_rspecifier = po.GetArg(4);
    // std::string fsts_wspecifier = po.GetArg(5);

    // ContextDependency ctx_dep;  // the tree.
    // ReadKaldiObject(tree_rxfilename, &ctx_dep);

    // TransitionModel trans_model;
    // ReadKaldiObject(model_rxfilename, &trans_model);

    // need VectorFst because we will change it by adding subseq symbol.
    // VectorFst<StdArc> *lex_fst = fst::ReadFstKaldi(lex_rxfilename);

    // std::vector<int32> disambig_syms;
    // if (disambig_rxfilename != "")
    //   if (!ReadIntegerVectorSimple(disambig_rxfilename, &disambig_syms))
    //     KALDI_ERR << "fstcomposecontext: Could not read disambiguation symbols from "
    //               << disambig_rxfilename;
    
    TrainingGraphCompiler gc(trans_model, ctx_dep, lex_fst, disambig_syms, gopts);

    // lex_fst = NULL;  // we gave ownership to gc.

    // SequentialInt32VectorReader transcript_reader(transcript_rspecifier);
    // TableWriter<fst::VectorFstHolder> fst_writer(fsts_wspecifier);

    int num_succeed = 0, num_fail = 0;

    // if (batch_size == 1) {  // We treat batch_size of 1 as a special case in order
      // to test more parts of the code.
      // for (; !transcript_reader.Done(); transcript_reader.Next()) {
        std::string key = id;
        // const std::vector<int32> &transcript = transcript_reader.Value();
        // VectorFst<StdArc> decode_fst;

        if (!gc.CompileGraphFromText(transcript, decode_fst)) {
          decode_fst->DeleteStates();  // Just make it empty.
        }
        if (decode_fst->Start() != fst::kNoStateId) {
          num_succeed++;
          return 0;
        }else {
          KALDI_WARN << "Empty decoding graph for utterance "
                     << key;
          return -1;
        }
  
    KALDI_LOG << "compile-train-graphs: succeeded for " << num_succeed
              << " graphs, failed for " << num_fail;
    return (num_succeed != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

};

#ifndef KALDI_ONLINE2LIBS_H_
#define KALDI_ONLINE2LIBS_H_

#include "base/kaldi-common.h"
#include "online2/online-ivector-feature.h"

namespace kaldi {


int ivector_extract_online2(const Matrix<BaseFloat> &feats,
                            OnlineIvectorExtractionConfig &ivector_config,
                            const std::string &id,
                            Matrix<BaseFloat> &ivectors);


}; // namespace kaldi
#endif

#ifndef KALDI_FEATLIBS_H_
#define KALDI_FEATLIBS_H_

#include "feat/feature-fbank.h"
#include "feat/feature-functions.h"
#include "feat/feature-mfcc.h"
#include "feat/pitch-functions.h"
#include "feat/wave-reader.h"

namespace kaldi {

int compute_mfcc_feats(const WaveData &wave_data, const MfccOptions &mfcc_opts,
                       const std::string &id, Matrix<BaseFloat> &features);

int compute_kaldi_pitch_feats(const WaveData &wave_data,
                              const PitchExtractionOptions &pitch_opts,
                              const std::string &id,
                              Matrix<BaseFloat> &features);
int process_kaldi_pitch_feats(const Matrix<BaseFloat> &features,
                              const ProcessPitchOptions &process_opts,
                              const std::string &id,
                              Matrix<BaseFloat> &processed_feats);

int compute_cmvn_stats(const Matrix<float> &feats, const std::string &id,
                       Matrix<double> &stats);
int apply_cmvn(const Matrix<float> &feat, const Matrix<double> &cmvn_stats,
               bool norm_means, bool norm_vars, const std::string &id,
               Matrix<float> &norm_feat);
int add_deltas(const Matrix<float> &feats, const std::string &id,
               Matrix<float> &new_feats);
int paste_feats(const std::vector<Matrix<BaseFloat>> &feats,
                const std::string &id, Matrix<BaseFloat> &output);

}; // namespace kaldi
#endif

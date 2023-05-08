/**
 * @file WaveformInfo.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The implemention of waveform relative class.
 * @version 0.1
 * @date 2022-11-09
 *
 * @copyright Copyright (c) 2022
 *
 */
#include "WaveformInfo.hh"

namespace ista {
ModeTransIndex WaveformInfo::getIndex(AnalysisMode analysis_mode,
                                      TransType trans_type) {
  ModeTransIndex model_trans_index = ModeTransIndex::kMaxRise;
  if ((analysis_mode == AnalysisMode::kMax) &&
      (trans_type == TransType::kFall)) {
    model_trans_index = ModeTransIndex::kMaxFall;
  } else if ((analysis_mode == AnalysisMode::kMin) &&
             (trans_type == TransType::kRise)) {
    model_trans_index = ModeTransIndex::kMinRise;
  } else if ((analysis_mode == AnalysisMode::kMin) &&
             (trans_type == TransType::kFall)) {
    model_trans_index = ModeTransIndex::kMinFall;
  }
  return model_trans_index;
}
}  // namespace ista

// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file WaveformInfo.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The implemention of waveform relative class.
 * @version 0.1
 * @date 2022-11-09
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

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
 * @file WaveformInfo.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The class of waveform.
 * @version 0.1
 * @date 2022-11-09
 */
#pragma once

#include <Eigen/Core>
#include <map>

#include "Type.hh"

namespace ista {

/**
 * @brief The waveform of node.
 *
 */
class Waveform {
 public:
  Waveform() = default;
  Waveform(double step_time_ns, const Eigen::VectorXd& waveform_vector)
      : _step_time_ns(step_time_ns), _waveform_vector(waveform_vector) {}
  ~Waveform() = default;

  Waveform(const Waveform& origin) = default;
  Waveform& operator=(const Waveform& origin) = default;

  Waveform(Waveform&& other) = default;
  Waveform& operator=(Waveform&& other) = default;

  double get_step_time_ns() const { return _step_time_ns; }

  Eigen::VectorXd& get_waveform_vector() { return _waveform_vector; }

 private:
  double _step_time_ns = 0.0;
  Eigen::VectorXd _waveform_vector;
};

/**
 * @brief The net propagated waveform info.
 *
 */
class WaveformInfo {
 public:
  ModeTransIndex getIndex(AnalysisMode analysis_mode, TransType trans_type);
  void addWaveform(Eigen::VectorXd waveform_vec, double step_time_ns,
                   AnalysisMode analysis_mode, TransType trans_type) {
    auto model_trans_index = getIndex(analysis_mode, trans_type);
    Waveform waveform(step_time_ns, waveform_vec);
    _waveforms.emplace(model_trans_index, std::move(waveform));
  }
  auto& getWaveform(AnalysisMode analysis_mode, TransType trans_type) {
    auto model_trans_index = getIndex(analysis_mode, trans_type);
    return _waveforms[model_trans_index];
  }

 private:
  std::map<ModeTransIndex, Waveform> _waveforms;
};

}  // namespace ista

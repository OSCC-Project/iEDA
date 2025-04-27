// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of
// Sciences Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan
// PSL v2. You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file Type.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief
 * @version 0.1
 * @date 2021-02-22
 */

#pragma once

#include <cmath>
#include <utility>

namespace ista {

enum class AnalysisMode : int { kMax = 1, kMin = 2, kMaxMin = 3 };
enum class TransType : int { kRise = 1, kFall = 2, kRiseFall = 3 };

#define IS_MAX(analysis_mode) (static_cast<int>(analysis_mode) & 0b01)
#define IS_MIN(analysis_mode) (static_cast<int>(analysis_mode) & 0b10)

#define IS_RISE(trans_type) (static_cast<int>(trans_type) & 0b01)
#define IS_FALL(trans_type) (static_cast<int>(trans_type) & 0b10)

#define FLIP_TRANS(type) \
  (((type) == TransType::kRise) ? TransType::kFall : TransType::kRise)

#define MODE_TRANS_SPLIT 4
#define MODE_SPLIT 2
#define TRANS_SPLIT 2

static constexpr int g_ns2ps = 1000;
static constexpr int g_ns2fs = 1000000;
static constexpr int g_ps2fs = 1000;
static constexpr int g_pf2ff = 1000;
static constexpr double g_pf2f = 1e-12;

static constexpr std::initializer_list<std::pair<AnalysisMode, TransType>>
    g_split_trans = {{AnalysisMode::kMax, TransType::kRise},
                     {AnalysisMode::kMax, TransType::kFall},
                     {AnalysisMode::kMin, TransType::kRise},
                     {AnalysisMode::kMin, TransType::kFall}};

static constexpr std::initializer_list<AnalysisMode> g_split_mode = {
    AnalysisMode::kMax, AnalysisMode::kMin};

enum class ModeTransIndex : int {
  kMaxRise = 0,
  kMaxFall = 1,
  kMinRise = 2,
  kMinFall = 3
};

ModeTransIndex mapToModeTransIndex(AnalysisMode mode, TransType type);

using ModeTransPair = std::pair<AnalysisMode, TransType>;

#define FOREACH_MODE_TRANS(mode, trans) for (auto [mode, trans] : g_split_trans)
#define FOREACH_MODE(mode) for (auto mode : g_split_mode)

#define NS_TO_FS(delay) ((delay) * static_cast<int64_t>(g_ns2fs))
#define FS_TO_NS(delay) ((delay) / static_cast<double>(g_ns2fs))
#define NS_TO_PS(delay) ((delay) * static_cast<int64_t>(g_ns2ps))
#define PS_TO_NS(delay) ((delay) / static_cast<double>(g_ns2ps))
#define PS_TO_FS(delay) ((delay) * static_cast<int64_t>(g_ps2fs))
#define FS_TO_PS(delay) ((delay) / static_cast<double>(g_ps2fs))

#define PF_TO_FF(cap) (static_cast<int>(std::ceil((cap)*g_pf2ff)))
#define FF_TO_PF(cap) ((cap) / static_cast<double>(g_pf2ff))

#define PF_TO_F(cap) ((cap)*g_pf2f)
#define F_TO_PF(cap) ((cap) / g_pf2f)

enum class DelayCalcMethod : int { kElmore = 0, kArnoldi = 1 };
enum class PropagationMethod : int { kDFS = 0, kBFS = 1 };

enum class CapacitiveUnit { kPF = 0, kFF = 1, kF = 2 };
enum class ResistanceUnit { kOHM = 0, kkOHM = 1 };
enum class TimeUnit { kNS = 0, kPS = 1, kFS = 2 };

inline double ConvertCapUnit(CapacitiveUnit src_unit, CapacitiveUnit snk_unit,
                             double cap) {
  double converted_cap = cap;
  if (src_unit != snk_unit) {
    if ((src_unit == CapacitiveUnit::kFF) &&
        (snk_unit == CapacitiveUnit::kPF)) {
      converted_cap = FF_TO_PF(cap);
    } else if ((src_unit == CapacitiveUnit::kPF) &&
               (snk_unit == CapacitiveUnit::kFF)) {
      converted_cap = PF_TO_FF(cap);
    }
  }

  return converted_cap;
}

constexpr double double_precesion = 1e-15;
constexpr bool IsDoubleEqual(double data1, double data2,
                             double eplison = double_precesion) {
  return std::abs(data1 - data2) < eplison;
}

// helper type for the std visitor
template <class... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

// Disallow the copy constructor and operator= functions.
#define FORBIDDEN_COPY(class_name)        \
  class_name(const class_name&) = delete; \
  void operator=(const class_name&) = delete

}  // namespace ista

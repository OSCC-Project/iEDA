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
 * @file PwrType.hh
 * @author shaozheqing (707005020@qq.com)
 * @brief The type for all ipower file.
 * @version 0.1
 * @date 2023-02-13
 */
#pragma once
namespace ipower {
enum class ScaleUnit { kSecond, kMS, kUS, kNS, kPS, kFS };
enum class PwrDataSource {
  kDefault,
  kAnnotate,
  kDataPropagation,
  kClockPropagation
};
enum class TricolorMark { kWhite, kGrey, kBlack };
enum class SeqPortType { kInput, kOutput };
enum class PwrAnalysisMode { kAveraged, kTimeBase, kClockCycle };

template <class... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

constexpr double CalcAveragePower(double rise_power, double fall_power) {
  return (rise_power + fall_power) / 2.0;
}
constexpr double HalfToggle(double toggle) { return toggle / 2.0; }
constexpr double CalcPercentage(double percentage) { return percentage * 100; }

static constexpr int g_nw2mw = 1000000;
static constexpr int g_nw2w = 1000000000;
static constexpr int g_mw2w = 1000;

#define NW_TO_MW(power) ((power) / static_cast<double>(g_nw2mw))
#define NW_TO_W(power) ((power) / static_cast<double>(g_nw2w))
#define MW_TO_W(power) ((power) / static_cast<double>(g_mw2w))

}  // namespace ipower

namespace ista {}
using namespace ista;

// helper type for the std visitor

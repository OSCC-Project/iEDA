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
 * @file FastStaDmpCeffSolver.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief DMP effective-capacitance solver state and equation declarations for CTS fast STA.
 */

#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <optional>
#include <tuple>
#include <utility>

#include "FastStaLibertyModel.hh"
#include "timing/FastStaClockTiming.hh"

namespace icts {

struct FastStaPiModel;

}  // namespace icts

namespace icts::fast_sta_dmp {

constexpr double kOhmPfToNs = 1e-3;
constexpr double kCapDeltaPf = 1e-3;
constexpr double kSmallRdNsPerPf = 1e-5;
constexpr double kEpsilon = 1e-18;
constexpr double kDriverParamTolerance = 0.01;
constexpr double kThresholdTimeTolerance = 0.01;
constexpr std::size_t kFindRootMaxIter = 20U;
constexpr std::size_t kNewtonRaphsonMaxIter = 100U;
constexpr std::size_t kMaxOrder = 3U;

enum class DmpParam : std::size_t
{
  kT0 = 0U,
  kDt = 1U,
  kCeff = 2U
};

enum class DmpFunc : std::size_t
{
  kY20 = 0U,
  kY50 = 1U,
  kIpi = 2U
};

constexpr auto ToIndex(DmpParam param) -> std::size_t
{
  return static_cast<std::size_t>(param);
}

constexpr auto ToIndex(DmpFunc func) -> std::size_t
{
  return static_cast<std::size_t>(func);
}

struct GateValues
{
  bool valid = false;
  double delay_ns = 0.0;
  double table_slew_ns = 0.0;
  double measured_slew_ns = 0.0;
  double t_vl_ns = 0.0;
};

using RootFunc = std::function<void(double, double&, double&)>;

auto OutputThreshold(const FastStaLibertyCell& cell, FastStaTransition transition) -> double;
auto InputThreshold(const FastStaLibertyCell& cell, FastStaTransition transition) -> double;
auto SlewLowerThreshold(const FastStaLibertyCell& cell, FastStaTransition transition) -> double;
auto SlewUpperThreshold(const FastStaLibertyCell& cell, FastStaTransition transition) -> double;
auto SlewDerate(const FastStaLibertyCell& cell) -> double;
auto GateDelaySlew(const FastStaLibertyCell& cell, FastStaTransition transition, double input_slew_ns, double ceff_pf)
    -> std::pair<double, double>;
auto GateModelRdNsPerPf(const FastStaLibertyCell& cell, const FastStaPiModel& pi, FastStaTransition transition, double input_slew_ns)
    -> double;
auto DmpExp(double x) -> double;
auto FindRoot(const RootFunc& func, double x1, double x2, double x_tolerance, std::size_t max_iter) -> std::optional<double>;

class DmpSolver
{
 public:
  DmpSolver(const FastStaLibertyCell& cell, const FastStaPiModel& pi, FastStaTransition transition, double input_slew_ns);

  auto solve() -> FastStaDmpDriverResult;

 private:
  auto useCapAlgorithm() const -> bool;
  auto makeResult(FastStaDmpAlgorithm algorithm, double ceff_pf, double delay_ns, double slew_ns, bool waveform_valid,
                  double waveform_delay_ns = 0.0) const -> FastStaDmpDriverResult;
  auto solveCap() -> FastStaDmpDriverResult;
  auto solvePi() -> FastStaDmpDriverResult;
  auto solveZeroC2() -> FastStaDmpDriverResult;
  auto initPi() -> bool;
  auto initZeroC2() -> bool;
  auto findDriverParams(double ceff_seed_pf) -> bool;
  auto gateDelays(double ceff_pf) const -> GateValues;
  auto y(double t_ns, double t0_ns, double dt_ns, double cl_pf) const -> std::pair<double, double>;
  auto y0(double t_ns, double cl_pf) const -> double;
  auto y0dt(double t_ns, double cl_pf) const -> double;
  auto y0dcl(double t_ns, double cl_pf) const -> double;
  auto dy(double t_ns, double t0_ns, double dt_ns, double cl_pf) const -> std::tuple<double, double, double>;
  auto evalDmpEquations() -> bool;
  auto evalPiEquations() -> bool;
  auto evalOnePoleEquations() -> bool;
  auto ipiIceff(double dt_ns, double ceff_time_ns, double ceff_pf) const -> double;
  auto newtonRaphson() -> bool;
  auto luDecomp() -> bool;
  auto luSolve() -> bool;
  auto findDriverDelaySlew() -> std::optional<std::pair<double, double>>;
  auto findVoCrossing(double threshold, double lower, double upper) -> std::optional<double>;
  auto vo(double t_ns) const -> std::pair<double, double>;
  auto v0(double t_ns) const -> std::pair<double, double>;
  auto voCrossingUpperBound() const -> double;

  static auto allFinite(const std::array<double, kMaxOrder>& values) -> bool;
  static auto allFinite(const std::array<std::array<double, kMaxOrder>, kMaxOrder>& values) -> bool;

  const FastStaLibertyCell* _cell = nullptr;
  FastStaTransition _transition = FastStaTransition::kRise;
  double _input_slew_ns = 0.0;
  double _near_cap_pf = 0.0;
  double _far_cap_pf = 0.0;
  double _rpi_ns_per_pf = 0.0;
  double _rd_ns_per_pf = 0.0;
  double _output_threshold = 0.5;
  double _slew_lower_threshold = 0.3;
  double _slew_upper_threshold = 0.7;
  double _slew_derate = 1.0;
  FastStaDmpAlgorithm _algorithm = FastStaDmpAlgorithm::kCap;
  std::size_t _nr_order = 1U;
  double _t0_ns = 0.0;
  double _dt_ns = 0.0;
  double _ceff_pf = 0.0;
  std::array<double, kMaxOrder> _x{};
  std::array<double, kMaxOrder> _fvec{};
  std::array<std::array<double, kMaxOrder>, kMaxOrder> _fjac{};
  std::array<double, kMaxOrder> _scale{};
  std::array<double, kMaxOrder> _p{};
  std::array<std::size_t, kMaxOrder> _index{};
  double _p1_per_ns = 0.0;
  double _p2_per_ns = 0.0;
  double _z1_per_ns = 0.0;
  double _k0 = 0.0;
  double _k1 = 0.0;
  double _k2 = 0.0;
  double _k3 = 0.0;
  double _k4 = 0.0;
  double _a_coeff = 0.0;
  double _b_coeff = 0.0;
  double _d_coeff = 0.0;
};

}  // namespace icts::fast_sta_dmp

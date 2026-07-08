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
 * @file FastStaDmpCeffSolver.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief DMP effective-capacitance solver setup for CTS fast STA.
 */

#include "FastStaDmpCeffSolver.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <string>
#include <utility>

#include "FastStaLibertyModel.hh"
#include "clock_net_parasitic/FastStaClockNetParasitic.hh"
#include "timing/FastStaClockTiming.hh"

namespace icts::fast_sta_dmp {

DmpSolver::DmpSolver(const FastStaLibertyCell& cell, const FastStaPiModel& pi, FastStaTransition transition, double input_slew_ns)
    : _cell(&cell),
      _transition(transition),
      _input_slew_ns(std::max(0.0, input_slew_ns)),
      _near_cap_pf(std::max(0.0, pi.near_cap_pf)),
      _far_cap_pf(std::max(0.0, pi.far_cap_pf)),
      _rpi_ns_per_pf(std::max(0.0, pi.resistance_ohm) * kOhmPfToNs),
      _rd_ns_per_pf(GateModelRdNsPerPf(cell, pi, transition, _input_slew_ns)),
      _output_threshold(OutputThreshold(cell, transition)),
      _slew_lower_threshold(SlewLowerThreshold(cell, transition)),
      _slew_upper_threshold(SlewUpperThreshold(cell, transition)),
      _slew_derate(SlewDerate(cell))
{
}

auto DmpSolver::solve() -> FastStaDmpDriverResult
{
  if (useCapAlgorithm()) {
    return solveCap();
  }
  if (_near_cap_pf < _far_cap_pf * 1e-3) {
    return solveZeroC2();
  }
  return solvePi();
}

auto DmpSolver::useCapAlgorithm() const -> bool
{
  return _rd_ns_per_pf < kSmallRdNsPerPf || _rpi_ns_per_pf < _rd_ns_per_pf * 1e-3 || _far_cap_pf == 0.0 || _far_cap_pf < _near_cap_pf * 1e-3
         || _rpi_ns_per_pf == 0.0;
}

auto DmpSolver::makeResult(FastStaDmpAlgorithm algorithm, double ceff_pf, double delay_ns, double slew_ns, bool waveform_valid,
                           double waveform_delay_ns) const -> FastStaDmpDriverResult
{
  return FastStaDmpDriverResult{
      .valid = true,
      .driver_waveform_valid = waveform_valid,
      .algorithm = algorithm,
      .transition = _transition,
      .driver_cell_master = _cell->cell_master,
      .ceff_pf = std::max(0.0, ceff_pf),
      .gate_delay_ns = std::max(0.0, delay_ns),
      .driver_slew_ns = std::max(0.0, slew_ns),
      .driver_waveform_delay_ns = std::max(0.0, waveform_delay_ns),
      .t0_ns = _t0_ns,
      .dt_ns = std::max(0.0, _dt_ns),
      .near_cap_pf = _near_cap_pf,
      .far_cap_pf = _far_cap_pf,
      .rpi_ns_per_pf = _rpi_ns_per_pf,
      .rd_ns_per_pf = _rd_ns_per_pf,
      .input_threshold = InputThreshold(*_cell, _transition),
      .output_threshold = _output_threshold,
      .slew_lower_threshold = _slew_lower_threshold,
      .slew_upper_threshold = _slew_upper_threshold,
      .slew_derate = _slew_derate,
      .pole1_per_ns = _p1_per_ns,
      .pole2_per_ns = _p2_per_ns,
      .zero1_per_ns = _z1_per_ns,
      .k0 = _k0,
      .k1 = _k1,
      .k2 = _k2,
      .k3 = _k3,
      .k4 = _k4,
  };
}

auto DmpSolver::solveCap() -> FastStaDmpDriverResult
{
  const auto ceff_pf = _near_cap_pf + _far_cap_pf;
  const auto [delay_ns, slew_ns] = GateDelaySlew(*_cell, _transition, _input_slew_ns, ceff_pf);
  return makeResult(FastStaDmpAlgorithm::kCap, ceff_pf, delay_ns, slew_ns, false);
}

auto DmpSolver::solvePi() -> FastStaDmpDriverResult
{
  _algorithm = FastStaDmpAlgorithm::kPi;
  _nr_order = 3U;
  if (!initPi()) {
    return solveCap();
  }

  if (!findDriverParams(_near_cap_pf + _far_cap_pf) && !findDriverParams(_near_cap_pf)) {
    return solveCap();
  }
  _ceff_pf = _x.at(ToIndex(DmpParam::kCeff));
  const auto [table_delay_ns, table_slew_ns] = GateDelaySlew(*_cell, _transition, _input_slew_ns, _ceff_pf);
  const auto driver_crossing = findDriverDelaySlew();
  if (!driver_crossing.has_value()) {
    return makeResult(FastStaDmpAlgorithm::kPi, _ceff_pf, table_delay_ns, table_slew_ns, false);
  }
  const auto [waveform_delay_ns, waveform_slew_ns] = *driver_crossing;
  return makeResult(FastStaDmpAlgorithm::kPi, _ceff_pf, table_delay_ns, waveform_slew_ns, true, waveform_delay_ns);
}

auto DmpSolver::solveZeroC2() -> FastStaDmpDriverResult
{
  _algorithm = FastStaDmpAlgorithm::kZeroC2;
  _nr_order = 2U;
  _ceff_pf = _far_cap_pf;
  if (!initZeroC2() || !findDriverParams(_far_cap_pf)) {
    const auto [delay_ns, slew_ns] = GateDelaySlew(*_cell, _transition, _input_slew_ns, _far_cap_pf);
    return makeResult(FastStaDmpAlgorithm::kZeroC2, _far_cap_pf, delay_ns, slew_ns, false);
  }

  const auto driver_crossing = findDriverDelaySlew();
  if (!driver_crossing.has_value()) {
    const auto [delay_ns, slew_ns] = GateDelaySlew(*_cell, _transition, _input_slew_ns, _far_cap_pf);
    return makeResult(FastStaDmpAlgorithm::kZeroC2, _far_cap_pf, delay_ns, slew_ns, false);
  }
  const auto [waveform_delay_ns, waveform_slew_ns] = *driver_crossing;
  return makeResult(FastStaDmpAlgorithm::kZeroC2, _far_cap_pf, waveform_delay_ns, waveform_slew_ns, true, waveform_delay_ns);
}

auto DmpSolver::initPi() -> bool
{
  if (_near_cap_pf <= 0.0 || _far_cap_pf <= 0.0 || _rpi_ns_per_pf <= 0.0 || _rd_ns_per_pf <= 0.0) {
    return false;
  }
  _z1_per_ns = 1.0 / (_rpi_ns_per_pf * _far_cap_pf);
  _k0 = 1.0 / (_rd_ns_per_pf * _near_cap_pf);
  const auto a = _rpi_ns_per_pf * _rd_ns_per_pf * _far_cap_pf * _near_cap_pf;
  const auto b = _rd_ns_per_pf * (_far_cap_pf + _near_cap_pf) + _rpi_ns_per_pf * _far_cap_pf;
  const auto discriminant = b * b - 4.0 * a;
  if (a <= 0.0 || discriminant < 0.0) {
    return false;
  }
  const auto root = std::sqrt(discriminant);
  _p1_per_ns = (b + root) / (2.0 * a);
  _p2_per_ns = (b - root) / (2.0 * a);
  const auto p1p2 = _p1_per_ns * _p2_per_ns;
  if (std::abs(p1p2) <= kEpsilon || std::abs(_p2_per_ns - _p1_per_ns) <= kEpsilon) {
    return false;
  }

  _k2 = _z1_per_ns / p1p2;
  _k1 = (1.0 - _k2 * (_p1_per_ns + _p2_per_ns)) / p1p2;
  _k4 = (_k1 * _p1_per_ns + _k2) / (_p2_per_ns - _p1_per_ns);
  _k3 = -_k1 - _k4;

  const auto z = (_far_cap_pf + _near_cap_pf) / (_rpi_ns_per_pf * _far_cap_pf * _near_cap_pf);
  _a_coeff = z / p1p2;
  _b_coeff = (z - _p1_per_ns) / (_p1_per_ns * (_p1_per_ns - _p2_per_ns));
  _d_coeff = (z - _p2_per_ns) / (_p2_per_ns * (_p2_per_ns - _p1_per_ns));
  return std::isfinite(_a_coeff) && std::isfinite(_b_coeff) && std::isfinite(_d_coeff);
}

auto DmpSolver::initZeroC2() -> bool
{
  if (_far_cap_pf <= 0.0 || _rpi_ns_per_pf <= 0.0 || _rd_ns_per_pf <= 0.0) {
    return false;
  }
  _z1_per_ns = 1.0 / (_rpi_ns_per_pf * _far_cap_pf);
  _p1_per_ns = 1.0 / (_far_cap_pf * (_rd_ns_per_pf + _rpi_ns_per_pf));
  if (std::abs(_z1_per_ns) <= kEpsilon || std::abs(_p1_per_ns) <= kEpsilon) {
    return false;
  }
  _k0 = _p1_per_ns / _z1_per_ns;
  _k2 = 1.0 / _k0;
  _k1 = (_p1_per_ns - _z1_per_ns) / (_p1_per_ns * _p1_per_ns);
  _k3 = -_k1;
  _k4 = 0.0;
  return std::isfinite(_k0) && std::isfinite(_k1) && std::isfinite(_k2) && std::isfinite(_k3);
}

auto DmpSolver::findDriverParams(double ceff_seed_pf) -> bool
{
  if (_nr_order == 3U) {
    _x.at(ToIndex(DmpParam::kCeff)) = ceff_seed_pf;
  }
  const auto gate_values = gateDelays(ceff_seed_pf);
  if (!gate_values.valid) {
    return false;
  }
  const auto threshold_span = _slew_upper_threshold - _slew_lower_threshold;
  if (threshold_span <= kEpsilon || _rd_ns_per_pf <= 0.0 || ceff_seed_pf <= 0.0) {
    return false;
  }
  const auto dt_ns = gate_values.measured_slew_ns / threshold_span;
  if (dt_ns <= kEpsilon) {
    return false;
  }
  const auto t0_ns = gate_values.delay_ns + std::log(1.0 - _output_threshold) * _rd_ns_per_pf * ceff_seed_pf - _output_threshold * dt_ns;
  _x.at(ToIndex(DmpParam::kDt)) = dt_ns;
  _x.at(ToIndex(DmpParam::kT0)) = t0_ns;
  if (!newtonRaphson()) {
    return false;
  }
  _t0_ns = _x.at(ToIndex(DmpParam::kT0));
  _dt_ns = _x.at(ToIndex(DmpParam::kDt));
  if (_nr_order == 3U) {
    _ceff_pf = _x.at(ToIndex(DmpParam::kCeff));
  }
  return std::isfinite(_t0_ns) && std::isfinite(_dt_ns) && _dt_ns > 0.0 && _ceff_pf >= 0.0;
}

auto DmpSolver::gateDelays(double ceff_pf) const -> GateValues
{
  const auto [delay_ns, table_slew_ns] = GateDelaySlew(*_cell, _transition, _input_slew_ns, ceff_pf);
  const auto threshold_span = _slew_upper_threshold - _slew_lower_threshold;
  if (!std::isfinite(delay_ns) || !std::isfinite(table_slew_ns) || table_slew_ns <= 0.0 || threshold_span <= kEpsilon) {
    return GateValues{};
  }
  const auto measured_slew_ns = table_slew_ns * _slew_derate;
  return GateValues{.valid = true,
                    .delay_ns = delay_ns,
                    .table_slew_ns = table_slew_ns,
                    .measured_slew_ns = measured_slew_ns,
                    .t_vl_ns = delay_ns - measured_slew_ns * (_output_threshold - _slew_lower_threshold) / threshold_span};
}

}  // namespace icts::fast_sta_dmp

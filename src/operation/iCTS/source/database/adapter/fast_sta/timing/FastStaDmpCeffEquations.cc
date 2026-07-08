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
 * @file FastStaDmpCeffEquations.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief DMP effective-capacitance equation solver for CTS fast STA.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <tuple>
#include <utility>

#include "FastStaDmpCeffSolver.hh"
#include "timing/FastStaClockTiming.hh"

namespace icts::fast_sta_dmp {

auto DmpSolver::y(double t_ns, double t0_ns, double dt_ns, double cl_pf) const -> std::pair<double, double>
{
  const auto t1_ns = t_ns - t0_ns;
  if (t1_ns <= 0.0) {
    return {0.0, t1_ns};
  }
  if (dt_ns <= kEpsilon) {
    return {0.0, t1_ns};
  }
  if (t1_ns <= dt_ns) {
    return {y0(t1_ns, cl_pf) / dt_ns, t1_ns};
  }
  return {(y0(t1_ns, cl_pf) - y0(t1_ns - dt_ns, cl_pf)) / dt_ns, t1_ns};
}

auto DmpSolver::y0(double t_ns, double cl_pf) const -> double
{
  const auto tau_ns = _rd_ns_per_pf * cl_pf;
  if (tau_ns <= kEpsilon) {
    return t_ns;
  }
  return t_ns - tau_ns * (1.0 - DmpExp(-t_ns / tau_ns));
}

auto DmpSolver::y0dt(double t_ns, double cl_pf) const -> double
{
  const auto tau_ns = _rd_ns_per_pf * cl_pf;
  if (tau_ns <= kEpsilon) {
    return 1.0;
  }
  return 1.0 - DmpExp(-t_ns / tau_ns);
}

auto DmpSolver::y0dcl(double t_ns, double cl_pf) const -> double
{
  const auto tau_ns = _rd_ns_per_pf * cl_pf;
  if (tau_ns <= kEpsilon || cl_pf <= kEpsilon) {
    return 0.0;
  }
  return _rd_ns_per_pf * ((1.0 + t_ns / tau_ns) * DmpExp(-t_ns / tau_ns) - 1.0);
}

auto DmpSolver::dy(double t_ns, double t0_ns, double dt_ns, double cl_pf) const -> std::tuple<double, double, double>
{
  const auto t1_ns = t_ns - t0_ns;
  if (t1_ns <= 0.0 || dt_ns <= kEpsilon) {
    return {0.0, 0.0, 0.0};
  }
  if (t1_ns <= dt_ns) {
    return {-y0dt(t1_ns, cl_pf) / dt_ns, -y0(t1_ns, cl_pf) / (dt_ns * dt_ns), y0dcl(t1_ns, cl_pf) / dt_ns};
  }
  return {-(y0dt(t1_ns, cl_pf) - y0dt(t1_ns - dt_ns, cl_pf)) / dt_ns,
          -(y0(t1_ns, cl_pf) + y0(t1_ns - dt_ns, cl_pf)) / (dt_ns * dt_ns) + y0dt(t1_ns - dt_ns, cl_pf) / dt_ns,
          (y0dcl(t1_ns, cl_pf) - y0dcl(t1_ns - dt_ns, cl_pf)) / dt_ns};
}

auto DmpSolver::evalDmpEquations() -> bool
{
  if (_algorithm == FastStaDmpAlgorithm::kPi) {
    return evalPiEquations();
  }
  if (_algorithm == FastStaDmpAlgorithm::kZeroC2) {
    return evalOnePoleEquations();
  }
  return false;
}

auto DmpSolver::evalPiEquations() -> bool
{
  const auto t0_ns = _x.at(ToIndex(DmpParam::kT0));
  const auto dt_ns = _x.at(ToIndex(DmpParam::kDt));
  const auto ceff_pf = _x.at(ToIndex(DmpParam::kCeff));
  if (ceff_pf < 0.0 || ceff_pf > _far_cap_pf + _near_cap_pf || dt_ns <= 0.0) {
    return false;
  }

  const auto gate_values = gateDelays(ceff_pf);
  if (!gate_values.valid) {
    return false;
  }
  auto ceff_time_ns = gate_values.measured_slew_ns / (_slew_upper_threshold - _slew_lower_threshold);
  ceff_time_ns = std::min(ceff_time_ns, 1.4 * dt_ns);
  const auto exp_p1_dt = DmpExp(-_p1_per_ns * dt_ns);
  const auto exp_p2_dt = DmpExp(-_p2_per_ns * dt_ns);
  const auto tau_ns = _rd_ns_per_pf * ceff_pf;
  if (tau_ns <= kEpsilon) {
    return false;
  }
  const auto exp_dt_rd_ceff = DmpExp(-dt_ns / tau_ns);
  const auto y50 = y(gate_values.delay_ns, t0_ns, dt_ns, ceff_pf).first;
  const auto y20 = y(gate_values.t_vl_ns, t0_ns, dt_ns, ceff_pf).first;
  _fvec.at(ToIndex(DmpFunc::kIpi)) = ipiIceff(dt_ns, ceff_time_ns, ceff_pf);
  _fvec.at(ToIndex(DmpFunc::kY50)) = y50 - _output_threshold;
  _fvec.at(ToIndex(DmpFunc::kY20)) = y20 - _slew_lower_threshold;
  _fjac.at(ToIndex(DmpFunc::kIpi)).at(ToIndex(DmpParam::kT0)) = 0.0;
  _fjac.at(ToIndex(DmpFunc::kIpi)).at(ToIndex(DmpParam::kDt))
      = (-_a_coeff * dt_ns + _b_coeff * dt_ns * exp_p1_dt - (2.0 * _b_coeff / _p1_per_ns) * (1.0 - exp_p1_dt) + _d_coeff * dt_ns * exp_p2_dt
         - (2.0 * _d_coeff / _p2_per_ns) * (1.0 - exp_p2_dt)
         + _rd_ns_per_pf * ceff_pf * (dt_ns + dt_ns * exp_dt_rd_ceff - 2.0 * _rd_ns_per_pf * ceff_pf * (1.0 - exp_dt_rd_ceff)))
        / (_rd_ns_per_pf * dt_ns * dt_ns * dt_ns);
  _fjac.at(ToIndex(DmpFunc::kIpi)).at(ToIndex(DmpParam::kCeff))
      = (2.0 * _rd_ns_per_pf * ceff_pf - dt_ns - (2.0 * _rd_ns_per_pf * ceff_pf + dt_ns) * exp_dt_rd_ceff) / (dt_ns * dt_ns);

  std::tie(_fjac.at(ToIndex(DmpFunc::kY20)).at(ToIndex(DmpParam::kT0)), _fjac.at(ToIndex(DmpFunc::kY20)).at(ToIndex(DmpParam::kDt)),
           _fjac.at(ToIndex(DmpFunc::kY20)).at(ToIndex(DmpParam::kCeff))) = dy(gate_values.t_vl_ns, t0_ns, dt_ns, ceff_pf);
  std::tie(_fjac.at(ToIndex(DmpFunc::kY50)).at(ToIndex(DmpParam::kT0)), _fjac.at(ToIndex(DmpFunc::kY50)).at(ToIndex(DmpParam::kDt)),
           _fjac.at(ToIndex(DmpFunc::kY50)).at(ToIndex(DmpParam::kCeff))) = dy(gate_values.delay_ns, t0_ns, dt_ns, ceff_pf);
  return allFinite(_fvec) && allFinite(_fjac);
}

auto DmpSolver::evalOnePoleEquations() -> bool
{
  auto t0_ns = _x.at(ToIndex(DmpParam::kT0));
  auto dt_ns = _x.at(ToIndex(DmpParam::kDt));
  const auto gate_values = gateDelays(_ceff_pf);
  if (!gate_values.valid) {
    return false;
  }
  if (dt_ns <= 0.0) {
    dt_ns = std::max(std::abs(gate_values.delay_ns - gate_values.t_vl_ns) / 100.0, 1e-6);
    _x.at(ToIndex(DmpParam::kDt)) = dt_ns;
  }
  _fvec.at(ToIndex(DmpFunc::kY50)) = y(gate_values.delay_ns, t0_ns, dt_ns, _ceff_pf).first - _output_threshold;
  _fvec.at(ToIndex(DmpFunc::kY20)) = y(gate_values.t_vl_ns, t0_ns, dt_ns, _ceff_pf).first - _slew_lower_threshold;

  double unused = 0.0;
  std::tie(_fjac.at(ToIndex(DmpFunc::kY20)).at(ToIndex(DmpParam::kT0)), _fjac.at(ToIndex(DmpFunc::kY20)).at(ToIndex(DmpParam::kDt)), unused)
      = dy(gate_values.t_vl_ns, t0_ns, dt_ns, _ceff_pf);
  std::tie(_fjac.at(ToIndex(DmpFunc::kY50)).at(ToIndex(DmpParam::kT0)), _fjac.at(ToIndex(DmpFunc::kY50)).at(ToIndex(DmpParam::kDt)), unused)
      = dy(gate_values.delay_ns, t0_ns, dt_ns, _ceff_pf);
  return allFinite(_fvec) && allFinite(_fjac);
}

auto DmpSolver::ipiIceff(double dt_ns, double ceff_time_ns, double ceff_pf) const -> double
{
  const auto exp_p1 = DmpExp(-_p1_per_ns * ceff_time_ns);
  const auto exp_p2 = DmpExp(-_p2_per_ns * ceff_time_ns);
  const auto tau_ns = _rd_ns_per_pf * ceff_pf;
  if (tau_ns <= kEpsilon || ceff_time_ns <= kEpsilon || dt_ns <= kEpsilon) {
    return 0.0;
  }
  const auto exp_ceff = DmpExp(-ceff_time_ns / tau_ns);
  const auto ipi = (_a_coeff * ceff_time_ns + (_b_coeff / _p1_per_ns) * (1.0 - exp_p1) + (_d_coeff / _p2_per_ns) * (1.0 - exp_p2))
                   / (_rd_ns_per_pf * ceff_time_ns * dt_ns);
  const auto iceff = (tau_ns * ceff_time_ns - tau_ns * tau_ns * (1.0 - exp_ceff)) / (_rd_ns_per_pf * ceff_time_ns * dt_ns);
  return ipi - iceff;
}

auto DmpSolver::newtonRaphson() -> bool
{
  for (std::size_t iter = 0U; iter < kNewtonRaphsonMaxIter; ++iter) {
    if (!evalDmpEquations()) {
      return false;
    }
    for (std::size_t i = 0U; i < _nr_order; ++i) {
      _p.at(i) = -_fvec.at(i);
    }
    if (!luDecomp() || !luSolve()) {
      return false;
    }

    auto all_under_tolerance = true;
    for (std::size_t i = 0U; i < _nr_order; ++i) {
      if (std::abs(_p.at(i)) > std::max(std::abs(_x.at(i)), 1e-12) * kDriverParamTolerance) {
        all_under_tolerance = false;
      }
      _x.at(i) += _p.at(i);
    }
    if (all_under_tolerance) {
      return evalDmpEquations();
    }
  }
  return false;
}

auto DmpSolver::luDecomp() -> bool
{
  for (std::size_t i = 0U; i < _nr_order; ++i) {
    auto big = 0.0;
    for (std::size_t j = 0U; j < _nr_order; ++j) {
      big = std::max(big, std::abs(_fjac.at(i).at(j)));
    }
    if (big <= kEpsilon) {
      return false;
    }
    _scale.at(i) = 1.0 / big;
  }

  for (std::size_t j = 0U; j < _nr_order; ++j) {
    for (std::size_t i = 0U; i < j; ++i) {
      auto sum = _fjac.at(i).at(j);
      for (std::size_t k = 0U; k < i; ++k) {
        sum -= _fjac.at(i).at(k) * _fjac.at(k).at(j);
      }
      _fjac.at(i).at(j) = sum;
    }

    auto big = 0.0;
    auto imax = j;
    for (std::size_t i = j; i < _nr_order; ++i) {
      auto sum = _fjac.at(i).at(j);
      for (std::size_t k = 0U; k < j; ++k) {
        sum -= _fjac.at(i).at(k) * _fjac.at(k).at(j);
      }
      _fjac.at(i).at(j) = sum;
      const auto scaled = _scale.at(i) * std::abs(sum);
      if (scaled >= big) {
        big = scaled;
        imax = i;
      }
    }
    if (j != imax) {
      for (std::size_t k = 0U; k < _nr_order; ++k) {
        std::swap(_fjac.at(imax).at(k), _fjac.at(j).at(k));
      }
      _scale.at(imax) = _scale.at(j);
    }
    _index.at(j) = imax;
    if (std::abs(_fjac.at(j).at(j)) <= kEpsilon) {
      _fjac.at(j).at(j) = kEpsilon;
    }
    if (j != _nr_order - 1) {
      const auto pivot = 1.0 / _fjac.at(j).at(j);
      for (std::size_t i = j + 1U; i < _nr_order; ++i) {
        _fjac.at(i).at(j) *= pivot;
      }
    }
  }
  return true;
}

auto DmpSolver::luSolve() -> bool
{
  auto non_zero = kMaxOrder;
  for (std::size_t i = 0U; i < _nr_order; ++i) {
    const auto iperm = _index.at(i);
    if (iperm >= _nr_order) {
      return false;
    }
    auto sum = _p.at(iperm);
    _p.at(iperm) = _p.at(i);
    if (non_zero != kMaxOrder) {
      for (std::size_t j = non_zero; j < i; ++j) {
        sum -= _fjac.at(i).at(j) * _p.at(j);
      }
    } else if (sum != 0.0) {
      non_zero = i;
    }
    _p.at(i) = sum;
  }
  for (std::size_t i = _nr_order; i-- > 0U;) {
    auto sum = _p.at(i);
    for (std::size_t j = i + 1U; j < _nr_order; ++j) {
      sum -= _fjac.at(i).at(j) * _p.at(j);
    }
    if (std::abs(_fjac.at(i).at(i)) <= kEpsilon) {
      return false;
    }
    _p.at(i) = sum / _fjac.at(i).at(i);
  }
  return allFinite(_p);
}

auto DmpSolver::findDriverDelaySlew() -> std::optional<std::pair<double, double>>
{
  const auto upper = voCrossingUpperBound();
  const auto delay = findVoCrossing(_output_threshold, _t0_ns, upper);
  if (!delay.has_value()) {
    return std::nullopt;
  }
  const auto tl = findVoCrossing(_slew_lower_threshold, _t0_ns, *delay);
  const auto th = findVoCrossing(_slew_upper_threshold, *delay, upper);
  if (!tl.has_value() || !th.has_value()) {
    return std::nullopt;
  }
  return std::pair<double, double>{*delay, std::max(0.0, (*th - *tl) / _slew_derate)};
}

auto DmpSolver::findVoCrossing(double threshold, double lower, double upper) -> std::optional<double>
{
  if (upper <= lower) {
    upper = lower + std::max(_dt_ns, 1e-6);
  }
  return FindRoot(
      [&](double t, double& y_value, double& dy_value) -> void {
        const auto [vo_value, dvo_dt] = vo(t);
        y_value = vo_value - threshold;
        dy_value = dvo_dt;
      },
      lower, upper, kThresholdTimeTolerance, kFindRootMaxIter);
}

auto DmpSolver::vo(double t_ns) const -> std::pair<double, double>
{
  const auto t1_ns = t_ns - _t0_ns;
  if (t1_ns <= 0.0 || _dt_ns <= kEpsilon) {
    return {0.0, 0.0};
  }
  if (t1_ns <= _dt_ns) {
    const auto [value, deriv] = v0(t1_ns);
    return {value / _dt_ns, deriv / _dt_ns};
  }
  const auto [value, deriv] = v0(t1_ns);
  const auto [dt_value, dt_deriv] = v0(t1_ns - _dt_ns);
  return {(value - dt_value) / _dt_ns, (deriv - dt_deriv) / _dt_ns};
}

auto DmpSolver::v0(double t_ns) const -> std::pair<double, double>
{
  if (_algorithm == FastStaDmpAlgorithm::kPi) {
    const auto exp_p1 = DmpExp(-_p1_per_ns * t_ns);
    const auto exp_p2 = DmpExp(-_p2_per_ns * t_ns);
    return {_k0 * (_k1 + _k2 * t_ns + _k3 * exp_p1 + _k4 * exp_p2), _k0 * (_k2 - _k3 * _p1_per_ns * exp_p1 - _k4 * _p2_per_ns * exp_p2)};
  }
  if (_algorithm == FastStaDmpAlgorithm::kZeroC2) {
    const auto exp_p1 = DmpExp(-_p1_per_ns * t_ns);
    return {_k0 * (_k1 + _k2 * t_ns + _k3 * exp_p1), _k0 * (_k2 - _k3 * _p1_per_ns * exp_p1)};
  }
  return {0.0, 0.0};
}

auto DmpSolver::voCrossingUpperBound() const -> double
{
  if (_algorithm == FastStaDmpAlgorithm::kPi) {
    return _t0_ns + _dt_ns + (_far_cap_pf + _near_cap_pf) * (_rd_ns_per_pf + _rpi_ns_per_pf) * 2.0;
  }
  if (_algorithm == FastStaDmpAlgorithm::kZeroC2) {
    return _t0_ns + _dt_ns + _far_cap_pf * (_rd_ns_per_pf + _rpi_ns_per_pf) * 2.0;
  }
  return _t0_ns + _dt_ns;
}

auto DmpSolver::allFinite(const std::array<double, kMaxOrder>& values) -> bool
{
  for (const auto& value : values) {
    if (!std::isfinite(value)) {
      return false;
    }
  }
  return true;
}

auto DmpSolver::allFinite(const std::array<std::array<double, kMaxOrder>, kMaxOrder>& values) -> bool
{
  for (const auto& row : values) {
    for (const auto value : row) {
      if (!std::isfinite(value)) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace icts::fast_sta_dmp

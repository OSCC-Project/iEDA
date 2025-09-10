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
 * @file PwrArc.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The power arc class, include cell arc and net arc, mostly we consider
 * the cell arc for internal power and the net arc for switch power.
 * @version 0.1
 * @date 2023-01-18
 */
#pragma once

#include "PwrVertex.hh"
#include "liberty/Lib.hh"
#include "netlist/Net.hh"

namespace ipower {

/**
 * @brief The abstract class of power arc.
 *
 */
class PwrArc {
 public:
  PwrArc(PwrVertex* src, PwrVertex* snk) : _src(src), _snk(snk) {}
  virtual ~PwrArc() = default;
  virtual unsigned isInstArc() { return 0; }
  virtual unsigned isNetArc() { return 0; }

  auto* get_src() { return _src; }
  auto* get_snk() { return _snk; }

  unsigned exec(PwrFunc& the_power_func) { return the_power_func(this); }

 private:
  PwrVertex* _src;  //!< The arc src vertex.
  PwrVertex* _snk;  //!< The arc snk vertex.

  FORBIDDEN_COPY(PwrArc);
};

/**
 * @brief The instance power arc.
 *
 */
class PwrInstArc : public PwrArc {
 public:
  PwrInstArc(PwrVertex* src, PwrVertex* snk) : PwrArc(src, snk) {}
  ~PwrInstArc() override = default;

  unsigned isInstArc() override { return 1; }
  void set_power_arc_set(LibPowerArcSet* power_arc_set) {
    _power_arc_set = power_arc_set;
  }
  auto* get_power_arc_set() { return _power_arc_set; }

  void set_internal_power(double internal_power) {
    _internal_power = internal_power;
  }
  [[nodiscard]] double getInternalPower() const { return _internal_power.value_or(0.0); }
  [[nodiscard]] auto get_internal_power() const { return _internal_power; }

 private:
  LibPowerArcSet* _power_arc_set =
      nullptr;  //!< The cell internal power set of different when condition.

  std::optional<double> _internal_power; //!< The arc internal power.
};

/**
 * @brief The net power arc.
 *
 */
class PwrNetArc : public PwrArc {
 public:
  PwrNetArc(PwrVertex* src, PwrVertex* snk) : PwrArc(src, snk) {}
  ~PwrNetArc() override = default;
  unsigned isNetArc() override { return 1; }

  void set_net(Net* net) { _net = net; }
  auto* get_net() { return _net; }

 private:
  Net* _net;
};

}  // namespace ipower
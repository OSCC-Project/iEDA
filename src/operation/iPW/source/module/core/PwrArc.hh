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
#include "liberty/Liberty.hh"
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

  DISALLOW_COPY_AND_ASSIGN(PwrArc);
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
  void set_power_arc_set(LibertyPowerArcSet* power_arc_set) {
    _power_arc_set = power_arc_set;
  }
  auto* get_power_arc_set() { return _power_arc_set; }

 private:
  LibertyPowerArcSet* _power_arc_set =
      nullptr;  //!< The cell internal power set of different when condition.
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
  
  void set_net(Net* net) {_net = net;}
  auto* get_net() { return _net; }

 private:
  Net* _net;
};

}  // namespace ipower
/**
 * @file PwrCell.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The power cell, mapping to design inst, used for leakage power
 * analysis.
 * @version 0.1
 * @date 2023-01-18
 */
#pragma once

#include "include/PwrConfig.hh"
#include "netlist/Instance.hh"

namespace ipower {

/**
 * @brief The power cell mapped to netlist instance.
 *
 */
class PwrCell {
 public:
  explicit PwrCell(Instance* design_inst) : _design_inst(design_inst) {}
  ~PwrCell() = default;

  auto* get_design_inst() { return _design_inst; }

 private:
  Instance* _design_inst;
};

}  // namespace ipower
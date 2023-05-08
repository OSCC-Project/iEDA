/**
 * @file PwrCalcSPData.hh
 * @author shaozheqing (707005020@qq.com)
 * @brief clac sp data.
 * @version 0.1
 * @date 2023-04-18
 *
 * @copyright Copyright (c) 2023
 *
 */
#pragma once

#include "core/PwrGraph.hh"
#include "include/PwrConfig.hh"
#include "liberty/Liberty.hh"
#include "netlist/Instance.hh"

namespace ipower {
/**
 * calc sp data
 */
class PwrCalcSPData : public PwrFunc {
 public:
  double getSPData(std::string_view port_name, Instance* inst);
  double calcSPData(LibertyExpr* expr, Instance* inst);
};
}  // namespace ipower
/**
 * @file PwrCalcToggleSP.hh
 * @author shaozheqing (707005020@qq.com)
 * @brief Calc cell output toggle and sp according to port function.
 * @version 0.1
 * @date 2023-03-27
 *
 * @copyright Copyright (c) 2023
 *
 */
#pragma once

#include "Vector.hh"
#include "core/PwrCell.hh"
#include "core/PwrFunc.hh"
#include "include/PwrType.hh"
#include "liberty/Liberty.hh"
#include "sta/StaVertex.hh"

namespace ipower {

/**
 * @brief struct for toggle and sp value.
 *
 */
struct PwrToggleSPValue {
  double _toggle_rate_value;
  double _sp_value;
};

using PwrToggleSPCalcFunc = std::function<PwrToggleSPValue(
    const std::vector<PwrToggleSPValue>& input_port_toggle_sp)>;

/**
 * @brief calc toggle sp of cell output.
 *
 */
class PwrCalcToggleSP : public PwrFunc {
 public:
  unsigned operator()(PwrCell* the_inst) override;
  unsigned operator()(PwrVertex* the_vertex) override;

 private:
  std::optional<PwrToggleSPCalcFunc> getCombineLogicProcessFunc(
      LibertyExpr::Operator op);

  PwrToggleSPValue getToggleSPData(std::string_view port_name,
                                   ieda::Vector<PwrVertex*>& input_vertexes,
                                   PwrDataSource data_source);

  PwrToggleSPValue calcToggleSP(LibertyExpr* expr,
                                ieda::Vector<PwrVertex*>& input_vertexes);

  unsigned calcToggleSP(LibertyCell* lib_cell,
                        ieda::Vector<PwrVertex*>& input_vertexes,
                        ieda::Vector<PwrVertex*>& output_vertexes);
};

}  // namespace ipower
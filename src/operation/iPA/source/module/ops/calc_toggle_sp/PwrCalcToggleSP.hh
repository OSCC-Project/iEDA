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
 * @file PwrCalcToggleSP.hh
 * @author shaozheqing (707005020@qq.com)
 * @brief Calc cell output toggle and sp according to port function.
 * @version 0.1
 * @date 2023-03-27
 */
#pragma once

#include "Vector.hh"
#include "core/PwrCell.hh"
#include "core/PwrFunc.hh"
#include "include/PwrType.hh"
#include "liberty/Lib.hh"
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
      RustLibertyExprOp op);

  PwrToggleSPValue getToggleSPData(std::string_view port_name,
                                   ieda::Vector<PwrVertex*>& input_vertexes,
                                   PwrDataSource data_source);

  PwrToggleSPValue calcToggleSP(RustLibertyExpr* expr,
                                ieda::Vector<PwrVertex*>& input_vertexes);

  unsigned calcToggleSP(LibCell* lib_cell,
                        ieda::Vector<PwrVertex*>& input_vertexes,
                        ieda::Vector<PwrVertex*>& output_vertexes);
};

}  // namespace ipower
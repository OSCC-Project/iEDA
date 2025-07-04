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
 * @file AnnotateToggleSP.hh
 * @author shaozheqing (707005020@qq.com)
 * @brief Annotate toggle and SP to power graph.
 * @version 0.1
 * @date 2023-01-11
 */
#pragma once

#include "AnnotateData.hh"
#include "core/PwrData.hh"
#include "core/PwrFunc.hh"
#include "core/PwrGraph.hh"
#include "include/PwrType.hh"
#include "netlist/Netlist.hh"

namespace ipower {
/**
 * @brief Annotate toggle and SP from annotate Data to Power Data.
 *
 */
class AnnotateToggleSP : public PwrFunc {
 public:
  AnnotateToggleSP() = default;
  ~AnnotateToggleSP() override = default;
  unsigned operator()(PwrGraph* the_graph) override;

  void set_annotate_db(AnnotateDB* annotate_db) { _annotate_db = annotate_db; }

 private:
  AnnotateDB* _annotate_db = nullptr;
};

}  // namespace ipower
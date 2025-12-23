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
 * @file AnnotateToggleSP.cc
 * @author shaozheqing (707005020@qq.com)
 * @brief annotate toggle and SP implemention.
 * @version 0.1
 * @date 2023-02-09
 */
#include "AnnotateToggleSP.hh"

namespace ipower {

/**
 * @brief calculate toggle and sp from annoate data which is based on VCD.
 *
 */
unsigned AnnotateToggleSP::operator()(PwrGraph* the_graph) {
  if (_annotate_db->get_top_instance()) {
    _annotate_db->calcInstancesTcSP();
    auto& tc_sp_db = _annotate_db->getTcSp();

    for (auto& signal_tc_sp : tc_sp_db) {
      auto& net_name = signal_tc_sp->get_signal_name();
      auto toggle_data = signal_tc_sp->get_toggle();
      auto sp_data = signal_tc_sp->get_sp();
      // find current driver vertex.
      std::string net_name_str{net_name.data(), net_name.size()};

      PwrVertex* driver_vertex = the_graph->getDriverVertex(net_name_str);

      if (driver_vertex == nullptr) {
        LOG_ERROR << "signal " << net_name << "'s driver vertex is not found in power graph.";
        continue;
      }

      driver_vertex->addData(toggle_data, sp_data, PwrDataSource::kAnnotate,
                             std::nullopt);
      // find snk vertexs.
      auto& snk_arcs = driver_vertex->get_snk_arcs();
      for (auto& snk_arc : snk_arcs) {
        auto* snk_vertex = snk_arc->get_snk();
        snk_vertex->addData(toggle_data, sp_data, PwrDataSource::kAnnotate,
                            std::nullopt);
      }
    }
  }

  return 1;
}

}  // namespace ipower
/**
 * @file AnnotateToggleSP.cc
 * @author shaozheqing (707005020@qq.com)
 * @brief annotate toggle and SP implemention.
 * @version 0.1
 * @date 2023-02-09
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "AnnotateToggleSP.hh"

namespace ipower {

/**
 * @brief calculate toggle and sp from annoate data which is based on VCD.
 *
 */
unsigned AnnotateToggleSP::operator()(PwrGraph* the_graph) {
  _annotate_db->calcInstancesTcSP();
  auto& tc_sp_db = _annotate_db->getTcSp();

  for (auto& signal_tc_sp : tc_sp_db) {
    auto& net_name = signal_tc_sp->get_signal_name();
    auto toggle_data = signal_tc_sp->get_toggle();
    auto sp_data = signal_tc_sp->get_sp();
    // find current driver vertex.
    std::string net_name_str{net_name.data(), net_name.size()};

    PwrVertex* driver_vertex = the_graph->getDriverVertex(net_name_str);

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

  return 1;
}

}  // namespace ipower
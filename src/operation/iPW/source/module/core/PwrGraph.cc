/**
 * @file PwrGraph.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The implemention of power graph.
 * @version 0.1
 * @date 2023-01-19
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "PwrGraph.hh"

namespace ipower {

/**
 * @brief get the net driver vertex.
 *
 * @param net_name
 * @return PwrVertex*
 */
PwrVertex* PwrGraph::getDriverVertex(const std::string& net_name) {
  Netlist* nl = get_sta_graph()->get_nl();
  Net* cur_net = nl->findNet(net_name.c_str());
  DesignObject* cur_obj = cur_net->getDriver();
  auto driver_sta_vertex = get_sta_graph()->findVertex(cur_obj);
  LOG_FATAL_IF(!driver_sta_vertex)
      << "net arc" << net_name << "'s driver vertex is not find";
  PwrVertex* driver_pwr_vertex = staToPwrVertex(*driver_sta_vertex);
  return driver_pwr_vertex;
}

/**
 * @brief get power vertex of obj.
 * 
 * @param obj 
 * @return PwrVertex* 
 */
PwrVertex* PwrGraph::getPowerVertex(DesignObject* obj) {
  auto* the_sta_graph = get_sta_graph();
  auto the_sta_vertex = the_sta_graph->findVertex(obj);
  LOG_FATAL_IF(!the_sta_vertex)
      << "sta vertex " << obj->getFullName() << " is not found.";
  auto* the_pwr_vertex = staToPwrVertex(*the_sta_vertex);
  return the_pwr_vertex;
}

}  // namespace ipower

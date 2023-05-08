/**
 * @file PwrDumpGraph.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The class for dump the power graph data.
 * @version 0.1
 * @date 2023-04-04
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include <yaml-cpp/yaml.h>
#include <sstream>

#include "core/PwrFunc.hh"
#include "core/PwrGraph.hh"

namespace ipower {

/**
 * @brief dump the power graph information.
 *
 */
class PwrDumpGraphYaml : public PwrFunc {
 public:
  unsigned operator()(PwrVertex* the_vertex) override;
  unsigned operator()(PwrArc* the_arc) override;
  unsigned operator()(PwrGraph* the_graph) override;

  void printText(const char* file_name);

 private:
  YAML::Node _node;
};

/**
 * @brief dump the power graphviz for debug.
 *
 */
class PwrDumpGraphViz : public PwrFunc {
 public:
  unsigned operator()(PwrArc* the_arc) override;

  void printText(const char* file_name);

 private:
  std::stringstream _ss;  //!< for print information to string stream.
};

}  // namespace ipower
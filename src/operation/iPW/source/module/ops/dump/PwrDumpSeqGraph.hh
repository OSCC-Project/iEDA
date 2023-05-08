/**
 * @file PwrDumpSeqGraph.hh
 * @author shaozheqing (707005020@qq.com)
 * @brief Dump the seq graph for debug.
 * @version 0.1
 * @date 2023-03-06
 */

#pragma once

#include <yaml-cpp/yaml.h>

#include "core/PwrFunc.hh"
#include "core/PwrSeqGraph.hh"

namespace ipower {
/**
 * @brief The class for dump seq graph in yaml text file for debug.
 *
 */
class PwrDumpSeqYaml : public PwrFunc {
 public:
  unsigned operator()(PwrSeqVertex* the_vertex) override;
  // unsigned operator()(PwrSeqArc* the_arc) override;
  // unsigned operator()(PwrSeqGraph* the_graph) override;

  void printText(const char* file_name);

 private:
  YAML::Node _node;
};

/**
 * @brief The class for dump seq graph in graphviz format for GUI show.
 *
 */
class PwrDumpSeqGraphViz : public PwrFunc {
 public:
  unsigned operator()(PwrSeqGraph* the_graph) override;
};
}  // namespace ipower
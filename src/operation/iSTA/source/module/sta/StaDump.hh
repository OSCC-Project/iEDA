/**
 * @file StaDump.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief Dump the Sta data for debug.
 * @version 0.1
 * @date 2021-04-22
 */
#pragma once

#include <yaml-cpp/yaml.h>

#include "StaFunc.hh"

namespace ista {

/**
 * @brief The class for dump sta data in yaml text file for debug.
 *
 */
class StaDumpYaml : public StaFunc {
 public:
  unsigned operator()(StaVertex* the_vertex) override;
  unsigned operator()(StaArc* the_arc) override;
  unsigned operator()(StaGraph* the_graph) override;

  void printText(const char* file_name);

 private:
  YAML::Node _node;
};

/**
 * @brief The class for dump sta data in graphviz format for GUI show.
 *
 */
class StaDumpGraphViz : public StaFunc {
 public:
  unsigned operator()(StaGraph* the_graph) override;
};

}  // namespace ista

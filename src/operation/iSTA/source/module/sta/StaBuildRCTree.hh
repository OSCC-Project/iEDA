/**
 * @file StaBuildRCTree.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The class of build rc tree.
 * @version 0.1
 * @date 2021-04-14
 */
#pragma once

#include <yaml-cpp/yaml.h>

#include <string>

#include "StaFunc.hh"

namespace ista {

/**
 * @brief The functor of build rc tree.
 *
 */
class StaBuildRCTree : public StaFunc {
 public:
  StaBuildRCTree() = default;
  StaBuildRCTree(std::string&& spef_file_name, DelayCalcMethod calc_method);
  ~StaBuildRCTree() override = default;

  unsigned operator()(StaGraph* the_graph) override;

  std::unique_ptr<RcNet> createRcNet(Net* net);
  DelayCalcMethod get_calc_method() { return _calc_method; }

  void printYaml(const spef::Net& spef_net);
  void printYamlText(const char* file_name);

 private:
  std::string _spef_file_name;
  DelayCalcMethod _calc_method =
      DelayCalcMethod::kElmore;  //!< The delay calc method selected.

  YAML::Node _top_node;  //!< Dump yaml node.
};

}  // namespace ista

/**
 * @file StaLevelization.hh
 * @author shy long (longshy@pcl.ac.cn)
 * @brief
 * @version 0.1
 * @date 2021-09-16
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once

#include "StaFunc.hh"

namespace ista {

class StaLevelization : public StaFunc {
 public:
  StaLevelization() = default;
  ~StaLevelization() override = default;

  unsigned operator()(StaVertex* the_vertex) override;
  unsigned operator()(StaArc* the_arc) override;

  unsigned operator()(StaGraph* the_graph) override;
};

}  // namespace ista

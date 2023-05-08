/**
 * @file StaConstPropagation.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The class of const propagation, for the const node(The pin connect to
 * Vdd, Gnd.), it don't need timing analysis, so we can exclude some node.
 * @version 0.1
 * @date 2021-03-04
 */

#pragma once

#include "StaFunc.hh"

namespace ista {

class StaConstPropagation : public StaFunc {
 public:
  unsigned operator()(StaGraph* the_graph);
};

}  // namespace ista
/**
 * @file StaCheck.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The timing check for combination loop, liberty integrity, timing
 * constrain missing and else etc.
 * @version 0.1
 * @date 2021-03-01
 */

#pragma once

#include "StaFunc.hh"
#include "StaGraph.hh"
#include "sta/StaVertex.hh"

namespace ista {

/**
 * @brief The combination loop check functor.
 *
 */
class StaCombLoopCheck : public StaFunc {
 public:
  virtual unsigned operator()(StaGraph* the_graph);

 private:
  // loop record
  void printAndBreakLoop(bool is_fwd);
  std::queue<StaVertex*> _loop_record;
};

}  // namespace ista

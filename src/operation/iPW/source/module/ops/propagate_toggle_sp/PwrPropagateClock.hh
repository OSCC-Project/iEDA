/**
 * @file PwrPropagateClock.hh
 * @author shaozheqing (707005020@qq.com)
 * @brief Propagate clock vertexes.
 * @version 0.1
 * @date 2023-04-15
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include "core/PwrFunc.hh"
#include "core/PwrGraph.hh"

namespace ipower {
/**
 * @brief Propagate clock vertexes.
 *
 */
class PwrPropagateClock : public PwrFunc {
 public:
  unsigned operator()(PwrVertex* the_vertex) override;
  unsigned operator()(PwrGraph* the_graph) override;
};
}  // namespace ipower
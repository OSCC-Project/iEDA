/**
 * @file PwrLevelizeSeq.hh
 * @author shaozheqing (707005020@qq.com)
 * @brief Ranking of sequential logic units.
 * @version 0.1
 * @date 2023-02-27
 */

#pragma once

#include "core/PwrFunc.hh"
#include "core/PwrSeqGraph.hh"

namespace ipower {

/**
 * @brief levelize the PwrLevelization by traveling power graph.
 *
 */
class PwrLevelizeSeq : public PwrFunc {
 public:
  unsigned operator()(PwrSeqVertex* the_vertex) override;
  unsigned operator()(PwrSeqGraph* the_graph) override;
};
}  // namespace ipower

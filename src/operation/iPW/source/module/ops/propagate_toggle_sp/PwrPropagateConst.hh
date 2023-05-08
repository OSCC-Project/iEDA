/**
 * @file PwrPropagateConst.hh
 * @author shaozheqing (707005020@qq.com)
 * @brief Propagate const vertexes from VDD or GND.
 * @version 0.1
 * @date 2023-03-27
 */

#pragma once

#include "core/PwrFunc.hh"
#include "core/PwrGraph.hh"
#include "core/PwrSeqGraph.hh"

namespace ipower {
/**
 * @brief Propagate const vertexes from VDD or GND.
 *
 */
class PwrPropagateConst : public PwrFunc {
 public:
  unsigned operator()(PwrVertex* the_vertex) override;
  unsigned operator()(PwrGraph* the_graph) override;
  void setTieCellFanout(PwrSeqGraph* the_seq_graph);

 private:
  void setConstStaToPwrVertex(StaVertex* const_sta_vertex,
                              PwrVertex* const_pwr_vertex);
  void setConstPwrToSeqVertex(PwrVertex* const_pwr_vertex);

};
}  // namespace ipower

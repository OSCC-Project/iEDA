/**
 * @file PwrPropagateToggleSP.hh
 * @author shaozheqing (707005020@qq.com)
 * @brief Propagate toggle and sp.
 * @version 0.1
 * @date 2023-04-06
 */

#pragma once

#include "core/PwrFunc.hh"
#include "core/PwrGraph.hh"
#include "core/PwrSeqGraph.hh"

/**
 * @brief struct for toggle and sp value in power data.
 *
 */
struct PwrToggleSPData {
  double _toggle_value;
  double _sp_value;
};

namespace ipower {
/**
 * @brief Propagate toggle and sp.
 *
 */
class PwrPropagateToggleSP : public PwrFunc {
 public:
  unsigned operator()(PwrGraph* the_graph) override;
  unsigned operator()(PwrVertex* the_vertex) override;

 private:
  PwrToggleSPData getToggleSPData(PwrVertex* the_vertex);
  PwrToggleSPData calcSeqDataOutToggleSP(
      PwrVertex* data_in_vertex, PwrVertex* data_out_vertex,
      const PwrToggleSPData& seq_data_in_toggle_sp);
};
}  // namespace ipower
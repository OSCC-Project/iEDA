/**
 * @file AnnotateToggleSP.hh
 * @author shaozheqing (707005020@qq.com)
 * @brief Annotate toggle and SP to power graph.
 * @version 0.1
 * @date 2023-01-11
 */
#pragma once

#include "AnnotateData.hh"
#include "core/PwrData.hh"
#include "core/PwrFunc.hh"
#include "core/PwrGraph.hh"
#include "include/PwrType.hh"
#include "netlist/Netlist.hh"

namespace ipower {
/**
 * @brief Annotate toggle and SP from annotate Data to Power Data.
 *
 */
class AnnotateToggleSP : public PwrFunc {
 public:
  AnnotateToggleSP() = default;
  ~AnnotateToggleSP() override = default;
  unsigned operator()(PwrGraph* the_graph) override;

  void set_annotate_db(AnnotateDB* annotate_db) { _annotate_db = annotate_db; }

 private:
  AnnotateDB* _annotate_db;
};

}  // namespace ipower
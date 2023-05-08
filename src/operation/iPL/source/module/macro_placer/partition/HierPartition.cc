#include "HierPartition.hh"

#include <map>

namespace ipl::imp {
void HierParttion::init()
{
  std::vector<FPInst*> std_cell_list = _mdb->get_design()->get_std_cell_list();
  std::vector<FPInst*> macro_list = _mdb->get_design()->get_macro_list();
  for (FPInst* std_cell : std_cell_list) {
    _top_module->add_inst(std_cell);
  }
  for (FPInst* macro : macro_list) {
    _top_module->add_inst(macro);
  }
}
}  // namespace ipl::imp
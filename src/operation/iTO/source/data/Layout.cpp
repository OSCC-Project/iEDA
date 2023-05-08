#include "Layout.h"
#include "IdbCellMaster.h"
#include "IdbDesign.h"
#include "IdbInstance.h"

namespace ito {
Layout::Layout(IdbDesign *idb_design) {
  IdbInstanceList      *idb_insts = idb_design->get_instance_list();
  vector<IdbInstance *> insts = idb_insts->get_instance_list();
  for (IdbInstance *idb_inst : insts) {
    Inst *inst = new Inst(idb_inst);
    _insts.push_back(inst);
  }
};

} // namespace ito

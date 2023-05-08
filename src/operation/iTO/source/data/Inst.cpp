#include "../../api/ids.hpp"
#include "Inst.h"
#include "IdbInstance.h"

namespace ito {

Inst::Inst(IdbInstance *inst) {
  IdbCellMaster *idb_master = inst->get_cell_master();
  _master = new Master(idb_master);
}

} // namespace ito
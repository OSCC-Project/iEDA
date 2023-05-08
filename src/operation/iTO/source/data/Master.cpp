#include "Master.h"
#include "IdbCellMaster.h"

namespace ito {
Master::Master(idb::IdbCellMaster *idb_master) {
  _width = idb_master->get_width();
  _height = idb_master->get_height();
}
} // namespace ito

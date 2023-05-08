#pragma once

#include "Master.h"
#include "ids.hpp"
#include <string>
#include <vector>

using idb::IdbCellMaster;
using idb::IdbInstance;
using ito::Master;

namespace ito {
class Inst {
 public:
  Inst() = default;
  Inst(IdbInstance *inst);
  ~Inst() = default;
  Inst(const Inst &inst) = delete;
  Inst(Inst &&inst) = delete;

  Master *get_master() { return _master; }

 protected:
  Master *_master = nullptr;
};

} // namespace ito
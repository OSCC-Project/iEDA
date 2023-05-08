#pragma once

#include "Inst.h"
#include "ids.hpp"

namespace ito {
using idb::IdbDesign;
using idb::IdbInstance;
using idb::IdbInstanceList;

class Layout {
 public:
  Layout() = default;
  Layout(IdbDesign *idb_design);
  ~Layout() = default;

  std::vector<Inst *> get_insts() { return _insts; }

 private:
  std::vector<Inst *> _insts;
};
} // namespace ito

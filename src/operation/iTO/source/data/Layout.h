// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
#pragma once

#include "IdbCellMaster.h"
#include "IdbDesign.h"
#include "IdbInstance.h"
#include "Inst.h"
namespace ito {
class Layout
{
 public:
  Layout() = default;
  Layout(idb::IdbDesign* idb_design)
  {
    idb::IdbInstanceList* idb_insts = idb_design->get_instance_list();
    vector<idb::IdbInstance*> insts = idb_insts->get_instance_list();
    for (idb::IdbInstance* idb_inst : insts) {
      Inst* inst = new Inst(idb_inst);
      _insts.push_back(inst);
    }
  }
  ~Layout() = default;

  std::vector<Inst*> get_insts() { return _insts; }

 private:
  std::vector<Inst*> _insts;
};
}  // namespace ito

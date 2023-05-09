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
#include "ipl_io.h"

#include "gui_io.h"

namespace igui {

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  bool GuiIO::autoRunPlacer() {
    return true;
    //    return iplf::tmInst->autoRunPlacer();
  }

  void GuiIO::updateInstanceInFastMode(std::vector<iplf::FileInstance>& file_inst_list) {
    return _gui_win->get_scene()->updateInstanceInFastMode(file_inst_list);
  }

  bool GuiIO::guiUpdateInstanceInFastMode(std::string directory, bool b_reset) {
    if (b_reset) {
      iplf::plInst->resetDpIndex();
    }
    if (iplf::plInst->readInstanceDataFromDirectory(directory)) {
      updateInstanceInFastMode(iplf::plInst->get_file_inst_list());
      return true;
    } else {
      return false;
    }

    return false;
  }

}  // namespace igui
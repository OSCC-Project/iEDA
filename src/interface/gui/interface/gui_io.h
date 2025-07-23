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
#include <map>
#include <string>
#include <vector>

#include "builder.h"
#include "file_cts.h"
#include "idrc_violation.h"
#include "mainwindow.h"
#include "vec_net.h"

#define guiInst (igui::GuiIO::getInstance())
using namespace idb;

namespace igui {
  class GuiIO {
   public:
    static GuiIO* getInstance() {
      if (!_instance) {
        _instance = new GuiIO;
      }
      return _instance;
    }
    /// getter
    MainWindow* get_main_window() { return _gui_win; }
    /// win operation
    void init();
    void startGUI(IdbBuilder* builder = nullptr, std::string type = "");
    int exec();
    void showWindow();
    void hideWindow();

    /// data operation-
    void readDB(std::vector<std::string> lef_paths, std::string def_path);
    void readDB(IdbBuilder* _builder);
    void readDrcDb(std::map<std::string, std::vector<idrc::DrcViolation*>>& drc_db, int max_num = -1);
    void readClockTreeDb(std::vector<iplf::CtsTreeNodeMap*>& node_list);
    void readGraphDb(std::map<int, ivec::VecNet> net_map);

    bool captureDesign(std::string path);

    /// iFP
    bool autoRunFloorplan();

    /// iPL
    bool autoRunPlacer();
    void updateInstanceInFastMode(std::vector<iplf::FileInstance>& file_inst_list);
    bool guiUpdateInstanceInFastMode(std::string directory = "", bool b_reset = false);

    /// iCTS
    bool autoRunCTS();

    /// iRouter
    bool autoRunRouter();

    /// iDRC
    bool autoRunDrcCheckDef();

    /// timer
    void timerStart(int ms = 1000);
    void timerStop();

   private:
    static GuiIO* _instance;

    MainWindow* _gui_win = nullptr;
    QApplication* _app   = nullptr;

    GuiIO() { }
    ~GuiIO() = default;
  };

}  // namespace igui

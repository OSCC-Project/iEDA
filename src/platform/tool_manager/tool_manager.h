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
/**
 * @File Name: tool_manager.h
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2022-03-17
 *
 */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <string>
#include <vector>

#include "lm_net.h"

namespace iplf {

#define tmInst ToolManager::getInstance()

class RtIO;
class FpIO;

class ToolManager
{
 public:
  static ToolManager* getInstance()
  {
    if (!_instance) {
      _instance = new ToolManager;
    }
    return _instance;
  }

  /// iDB
  bool idbStart(std::string config_path);
  bool idbSave(std::string name);

  // /// GUI
  void guiInit();
  void guiStart(std::string type = "");
  void guiShow();
  void guiHide();
  int guiExec();
  void guiTimerStart(int ms = 1000);
  void guiTimerStop();
  void guiReadDb();
  void guiShowDrc(std::string detail_drc_path = "", int max_num = 100000);
  void guiShowClockTree();
  void guiShowGraph(std::map<int, ilm::LmNet> graph);

  void guiCaptrueDesign(std::string path = "");
  /// Eval
  // int64_t evalTotalWL(const std::vector<eval::WLNet*>& net_list, const std::string& wl_type);
  // void estimateDelay(std::vector<eval::TimingNet*> timing_net_list, const char* sta_workspace_path, const char* sdc_file_path,
  //                    std::vector<const char*> lib_file_path_list);

  /// iFP
  //   bool autoRunFloorplan(std::string config = "");
  //   bool floorplanInit();

  /// iPL
  bool autoRunPlacer(std::string config = "", bool enableJsonOutput = false);
  bool runPlacerFiller(std::string config = "");
  bool runPlacerIncrementalFlow(std::string config);
  bool runPlacerIncrementalLegalization();
  bool checkLegality();
  bool reportPlacer();

  // iNO
  bool RunNOFixFanout(std::string config = "");

  /// iTO
  bool autoRunTO(std::string config = "");
  bool RunTODrv(std::string config = "");
  bool RunTODrvSpecialNet(std::string config = "", std::string net_name = "");
  bool RunTOHold(std::string config = "");
  bool RunTOSetup(std::string config = "");
  bool RunTOBuffering(std::string config = "", std::string net_name = "");

  /// iCTS
  bool autoRunCTS(std::string config = "", std::string work_dir = "");
  bool reportCTS(std::string path = "");
  /// iRT
  bool autoRunRouter(std::string config_file_path = "");

  /// iDRC
  bool autoRunDRC(std::string config = "", std::string path = "", bool has_init = false);
  bool readDrcDetailFromFile(std::string path = "");
  bool saveDrcDetailToFile(std::string path = "");

  /// iSTA
  bool autoRunSTA(std::string config = "");
  bool initSTA(std::string config = "");
  bool runSTA(std::string config = "");

  /// iPW
  bool autoRunPower(std::string config = "");

  bool buildClockTree(std::string config = "", std::string data_path = "");
  bool saveClockTree(std::string data_path);

 private:
  static ToolManager* _instance;
  ToolManager();
  ~ToolManager() = default;

  ///
};

}  // namespace iplf

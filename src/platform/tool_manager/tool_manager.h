#pragma once
/**
 * iEDA
 * Copyright (C) 2021  PCL
 *
 * This program is free software;
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @File Name: core.h
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

#include "TimingNet.hpp"
#include "WLNet.hpp"
#include "wirelength/WLFactory.hpp"

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

  /// Eval
  int64_t evalTotalWL(const std::vector<eval::WLNet*>& net_list, const std::string& wl_type);
  void estimateDelay(std::vector<eval::TimingNet*> timing_net_list, const char* sta_workspace_path, const char* sdc_file_path,
                     std::vector<const char*> lib_file_path_list);

  /// iFP
  //   bool autoRunFloorplan(std::string config = "");
  //   bool floorplanInit();

  /// iPL
  bool autoRunPlacer(std::string config = "");
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
  bool RunTOHold(std::string config = "");
  bool RunTOSetup(std::string config = "");

  /// iCTS
  bool autoRunCTS(std::string config = "");
  bool reportCTS(std::string path = "");
  /// iRT
  bool autoRunRouter(std::string config_file_path = "");

  /// iDRC
  bool autoRunDRC(std::string config = "", std::string path = "");
  bool readDrcDetailFromFile(std::string path = "");
  bool saveDrcDetailToFile(std::string path = "");

  /// iSTA
  bool autoRunSTA(std::string config = "");
  bool initSTA(std::string config = "");
  bool runSTA(std::string config = "");

  bool buildClockTree(std::string config = "", std::string data_path = "");
  bool saveClockTree(std::string data_path);

 private:
  static ToolManager* _instance;
  ToolManager();
  ~ToolManager() = default;

  ///
};

}  // namespace iplf

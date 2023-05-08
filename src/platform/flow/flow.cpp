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
 * @File Name: platform.cpp
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2022-04-15
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "flow.h"

#include "tcl_main.h"

namespace iplf {
Flow* Flow::_instance = nullptr;
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool Flow::initFlow(string flow_config)
{
  /// read flow config
  if (!PLFConfig::getInstance()->initConfig(flow_config)) {
    std::cout << "PLFConfig init failed." << std::endl;
  }

  /// init GUI
  // std::cout << "gui start." << std::endl;
  // tmInst->guiStart();
  // tmInst->guiShow();
  // tmInst->guiReadDb();

  return true;
}

void Flow::run(char* param)
{
  if (PLFConfig::getInstance()->is_run_tcl()) {
    runTcl(param);
  } else {
    runFlow();
  }
}

void Flow::runTcl(char* path)
{
  tcl::tcl_start(path);
}

void Flow::runFlow()
{
  /// init DB
  if (tmInst->idbStart(PLFConfig::getInstance()->get_idb_path())) {
  }

  /// run fp
  /// fp autorun depends on fp_config, which has to be verified
  // if (PLFConfig::getInstance()->is_run_floorplan()) {
  //   if (tmInst->autoRunFloorplan(PLFConfig::getInstance()->get_ifp_path())) { }
  // }

  /// run placer
  if (PLFConfig::getInstance()->is_run_placer()) {
    if (tmInst->autoRunPlacer(PLFConfig::getInstance()->get_ipl_path())) {
    }
  }

  /// run TO
  if (PLFConfig::getInstance()->is_run_to()) {
    if (tmInst->autoRunTO(PLFConfig::getInstance()->get_ito_path())) {
    }
  }

  /// run cts
  if (PLFConfig::getInstance()->is_run_cts()) {
    if (tmInst->autoRunCTS(PLFConfig::getInstance()->get_icts_path())) {
    }
  }

  /// run router
  if (PLFConfig::getInstance()->is_run_router()) {
    if (tmInst->autoRunRouter(PLFConfig::getInstance()->get_irt_path())) {
    }
  }

  /// run filler
  if (PLFConfig::getInstance()->is_run_placer()) {
    if (tmInst->runPlacerFiller(PLFConfig::getInstance()->get_ipl_path())) {
    }
  }

  /// run DRC
  if (PLFConfig::getInstance()->is_run_drc()) {
    if (tmInst->autoRunDRC(PLFConfig::getInstance()->get_idrc_path())) {
    }
  }

  /// run gui
  if (PLFConfig::getInstance()->is_run_gui()) {
    tmInst->guiStart();
  }
}

}  // namespace iplf

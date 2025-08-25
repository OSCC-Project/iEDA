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
#include "py_ipl.h"

#include "idm.h"
#include "ipl_io/ipl_io.h"
#include "tool_manager.h"
namespace python_interface {

bool placerAutoRun(const std::string& config)
{
  bool run_ok = iplf::tmInst->autoRunPlacer(config);
  return run_ok;
}

bool placerRunFiller(const std::string& config)
{
  bool run_ok = iplf::tmInst->runPlacerFiller(config);
  return run_ok;
}

bool placerIncrementalFlow(const std::string& config)
{
  bool run_ok = iplf::tmInst->runPlacerIncrementalFlow(config);
  return run_ok;
}

bool placerIncrementalLG()
{
  auto* inst = iplf::PlacerIO::getInstance();
  bool run_ok = inst->runIncrementalLegalization();
  return run_ok;
}

bool placerCheckLegality()
{
  auto* inst = iplf::PlacerIO::getInstance();
  bool is_legal = inst->checkLegality();
  return is_legal;
}

bool placerReport()
{
  auto* inst = iplf::PlacerIO::getInstance();
  bool run_ok = inst->reportPlacement();
  return run_ok;
}

bool placerAiRun(const std::string& config, const std::string& onnx_path, const std::string& normalization_path)
{
  bool run_ok = iplf::tmInst->runAiPlacer(config, onnx_path, normalization_path);
  return run_ok;
}


bool placerInit(const std::string& config)
{
  auto* inst = iplf::PlacerIO::getInstance();
  inst->initPlacer(config);
  return true;
}

bool placerDestroy()
{
  auto* inst = iplf::PlacerIO::getInstance();
  inst->destroyPlacer();
  return true;
}

bool placerRunMP()
{
  auto* inst = iplf::PlacerIO::getInstance();
  bool run_ok = inst->runMacroPlacement();
  return run_ok;
}

bool placerRunGP()
{
  auto* inst = iplf::PlacerIO::getInstance();
  bool run_ok = inst->runGlobalPlacement();
  return run_ok;
}

bool placerRunLG()
{
  auto* inst = iplf::PlacerIO::getInstance();
  bool run_ok = inst->runLegalization();
  return run_ok;
}

bool placerRunDP()
{
  auto* inst = iplf::PlacerIO::getInstance();
  inst->runDetailPlacement();
  return true;
}

}  // namespace python_interface
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
#include "idrc_io.h"

#include "builder.h"
#include "flow_config.h"
#include "idm.h"
#include "idrc_api.h"
#include "report_manager.h"

#ifdef USE_PROFILER
#include <gperftools/profiler.h>
#endif

namespace iplf {
DrcIO* DrcIO::_instance = nullptr;
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DrcIO::clear()
{
  for (auto& [drc_rule, drc_list] : _detail_drc) {
    for (auto* drc : drc_list) {
      if (drc != nullptr) {
        delete drc;
        drc = nullptr;
      }
    }

    drc_list.clear();
    std::vector<idrc::DrcViolation*>().swap(drc_list);
  }

  _detail_drc.clear();
}
bool DrcIO::runDRC(std::string config, std::string report_path)
{
  if (config.empty()) {
    /// set config path
    config = flowConfigInst->get_idrc_path();
  }

  flowConfigInst->set_status_stage("iDRC - Design Rule Check");
  ieda::Stats stats;

  // auto result_drc = idrc::DrcAPIInst.getCheckResult();
  // auto result_connectivity = checkConnnectivity();
  _detail_drc.clear();
  auto result_drc_detail = getDetailCheckResult();
  std::map<std::string, int> result_drc;
  //   std::tuple<bool, std::vector<std::string>, std::vector<std::string>, int> result_connectivity;

  auto result_connectivity = checkConnnectivity();

  for (auto [rule_name, drc_list] : result_drc_detail) {
    result_drc.insert(std::make_pair(rule_name, drc_list.size()));
  }

  flowConfigInst->add_status_runtime(stats.elapsedRunTime());
  flowConfigInst->set_status_memmory(stats.memoryDelta());

  rptInst->reportDRC(report_path, result_drc, result_connectivity);

  return true;
}

std::tuple<bool, std::vector<std::string>, std::vector<std::string>, int> DrcIO::checkConnnectivity()
{
  return dmInst->isAllNetConnected();
}

std::map<std::string, std::vector<idrc::DrcViolation*>> DrcIO::getDetailCheckResult(std::string path)
{
  if (!_detail_drc.empty()) {
    return _detail_drc;
  }

  if (path.empty()) {
    // _detail_drc = idrc::DrcAPIInst.getDetailCheckResult();
    get_def_drc();
  } else {
    /// get drc detail data from file
    readDrcFromFile(path);
  }

  return _detail_drc;
}

void DrcIO::get_def_drc()
{
#ifdef USE_PROFILER
  ProfilerStart("idrc.prof");
#endif
  idrc::DrcApi drc_api;
  drc_api.init();
  auto violations = drc_api.checkDef();
  for (auto [type, violation_list] : violations) {
    std::string name = idrc::GetViolationTypeName()(type);

    _detail_drc.insert(std::make_pair(name, violation_list));
  }
#ifdef USE_PROFILER
  ProfilerStop();
#endif
}

void DrcIO::set_detail_drc(std::map<std::string, std::vector<idrc::DrcViolation*>>& detail_drc)
{
  clear();
  _detail_drc = detail_drc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int32_t DrcIO::get_buffer_size()
{
  int32_t buffer_size = sizeof(DrcFileHeader);
  for (auto [rule_name, drc_list] : _detail_drc) {
    buffer_size = buffer_size + sizeof(DrcResultHeader) + drc_list.size() * sizeof(DrcDetailResult);
  }
  return buffer_size;
}

bool DrcIO::readDrcFromFile(std::string path)
{
  if (path.empty()) {
    return false;
  }

  FileDrcManager file(path, (int32_t) DrcDbId::kDrcDetailInfo);

  return file.readFile();
}

bool DrcIO::saveDrcToFile(std::string path)
{
  // _detail_drc.clear();

  if (path.empty()) {
    return false;
  }

  getDetailCheckResult();

  FileDrcManager file(path, (int32_t) DrcDbId::kDrcDetailInfo);

  return file.writeFile();
}

}  // namespace iplf

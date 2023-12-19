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

#include "DrcAPI.hpp"
#include "builder.h"
#include "flow_config.h"
#include "idm.h"
#include "idrc_api.h"
#include "idrc_violation_enum.h"
#include "report_manager.h"

namespace iplf {
DrcIO* DrcIO::_instance = nullptr;
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool DrcIO::runDRC(std::string config, std::string report_path)
{
  if (config.empty()) {
    /// set config path
    config = flowConfigInst->get_idrc_path();
  }

  flowConfigInst->set_status_stage("iDRC - Design Rule Check");
  ieda::Stats stats;

  auto result_drc = idrc::DrcAPIInst.getCheckResult();
  auto result_connectivity = checkConnnectivity();

  flowConfigInst->add_status_runtime(stats.elapsedRunTime());
  flowConfigInst->set_status_memmory(stats.memoryDelta());

  rptInst->reportDRC(report_path, result_drc, result_connectivity);

  return true;
}

std::tuple<bool, std::vector<std::string>, std::vector<std::string>, int> DrcIO::checkConnnectivity()
{
  return dmInst->isAllNetConnected();
}

std::map<std::string, std::vector<idrc::DrcViolationSpot*>> DrcIO::getDetailCheckResult(std::string path)
{
  _detail_drc.clear();

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
  auto* idb_layout = dmInst->get_idb_layout();
  auto* idb_layers = idb_layout->get_layers();
  idrc::DrcApi drc_api;
  drc_api.init();
  auto violations = drc_api.checkDef();
  for (auto [type, violation_list] : violations) {
    std::string name = DrcViolationTypeInst->get_type_name(type);

    std::vector<idrc::DrcViolationSpot*> spot_list;
    spot_list.reserve(violation_list.size());

    for (auto* violation : violation_list) {
      idrc::DrcViolationSpot* spot = new idrc::DrcViolationSpot();
      auto* layer = idb_layers->find_routing_layer(violation->get_layer_id());
      if (layer != nullptr) {
        spot->set_layer_name(layer->get_name());
      }
      spot->set_layer_id(violation->get_layer_id());
      auto net_list = violation->get_net_ids();
      spot->set_net_id(*net_list.begin());
      auto vio_type = type == idrc::ViolationEnumType::kViolationShort ? idrc::ViolationType::kShort : idrc::ViolationType::kRoutingSpacing;
      spot->set_vio_type(vio_type);
      auto* rect = static_cast<idrc::DrcViolationRect*>(violation);
      spot->setCoordinate(rect->get_llx(), rect->get_lly(), rect->get_urx(), rect->get_ury());

      spot_list.emplace_back(spot);
    }

    _detail_drc.insert(std::make_pair(name, spot_list));
  }
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
  _detail_drc.clear();

  if (path.empty()) {
    return false;
  }

  getDetailCheckResult();

  FileDrcManager file(path, (int32_t) DrcDbId::kDrcDetailInfo);

  return file.writeFile();
}

}  // namespace iplf

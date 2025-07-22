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

#include "DRCInterface.hpp"
#include "builder.h"
#include "feature_manager.h"
#include "file_drc.h"
#include "flow_config.h"
#include "idm.h"
#include "report_manager.h"

#ifdef USE_PROFILER
#include <gperftools/profiler.h>
#endif

namespace iplf {
DrcIO* DrcIO::_instance = nullptr;
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool DrcIO::runDRC(std::string config, std::string report_path, bool has_init)
{
  flowConfigInst->set_status_stage("iDRC - Design Rule Check");
  ieda::Stats stats;

  if (!has_init) {
    std::map<std::string, std::any> config_map;
    DRCI.initDRC(config_map, false);
  }
  DRCI.checkDef();
  DRCI.destroyDRC();

  flowConfigInst->add_status_runtime(stats.elapsedRunTime());
  flowConfigInst->set_status_memmory(stats.memoryDelta());

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
  if (path.empty()) {
    return false;
  }

  FileDrcManager file(path, (int32_t) DrcDbId::kDrcDetailInfo);

  return file.writeFile();
}

std::map<std::string, std::map<std::string, std::vector<ids::Violation>>>& DrcIO::getDetailCheckResult(std::string path)
{
  return featureInst->get_type_layer_violation_map();
}

}  // namespace iplf

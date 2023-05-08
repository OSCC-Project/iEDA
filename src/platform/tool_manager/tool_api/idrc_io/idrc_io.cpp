#include "idrc_io.h"

#include "DrcAPI.hpp"
#include "builder.h"
#include "flow_config.h"
#include "idm.h"
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
    _detail_drc = idrc::DrcAPIInst.getDetailCheckResult();
  } else {
    /// get drc detail data from file
    readDrcFromFile(path);
  }

  return _detail_drc;
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

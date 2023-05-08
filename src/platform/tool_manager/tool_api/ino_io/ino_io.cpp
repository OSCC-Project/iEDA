#include "ino_io.h"

#include "builder.h"
#include "flow_config.h"
#include "iNO/api/NoApi.hpp"
#include "idm.h"

namespace iplf {
NoIO* NoIO::_instance = nullptr;
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool NoIO::runNOFixFanout(std::string config)
{
  // if (config.empty()) {
  //   /// set config path
  //   config = flowConfigInst->get_ito_path();
  // }

  flowConfigInst->set_status_stage("iNO - FixFanout");

  ieda::Stats stats;

  /// set data config
  NoApiInst.initNO(config);
  /// reset lib & sdc
  resetConfig(NoApiInst.get_no_config());

  // ToApiInst.iTODataInit(dmInst->get_idb_builder(), nullptr);
  NoApiInst.iNODataInit(dmInst->get_idb_builder(), nullptr);
  NoApiInst.fixFanout();

  flowConfigInst->add_status_runtime(stats.elapsedRunTime());
  flowConfigInst->set_status_memmory(stats.memoryDelta());

  return true;
}

void NoIO::resetConfig(ino::NoConfig* no_config)
{
  if (no_config == nullptr) {
    return;
  }

  idm::DataConfig& db_config = dmInst->get_config();

  if (db_config.get_lib_paths().size() > 0) {
    no_config->set_lib_files(db_config.get_lib_paths());
  }

  if (!db_config.get_sdc_path().empty()) {
    no_config->set_sdc_file(db_config.get_sdc_path());
  }
}

}  // namespace iplf

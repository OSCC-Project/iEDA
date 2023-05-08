#include "ieval_io.h"

#include "TimingEval.hpp"
#include "TimingNet.hpp"
#include "WLNet.hpp"
#include "WirelengthEval.hpp"
#include "builder.h"
#include "idm.h"

namespace iplf {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int64_t EvalIO::evalTotalWL(const std::vector<eval::WLNet*>& net_list, const std::string& wl_type)
{
  eval::WirelengthEval eval;
  eval.set_net_list(net_list);
  return eval.evalTotalWL("kHPWL");
}

/// timing eval
void EvalIO::estimateDelay(std::vector<eval::TimingNet*> timing_net_list, const char* sta_workspace_path, const char* sdc_file_path,
                           std::vector<const char*> lib_file_path_list)
{
  idb::IdbBuilder* idb_builder = dmInst->get_idb_builder();
  eval::TimingEval eval;
  eval.set_timing_net_list(timing_net_list);
  eval.estimateDelay(idb_builder, sta_workspace_path, lib_file_path_list, sdc_file_path);
}

}  // namespace iplf

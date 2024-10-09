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
#include "ieval_io.h"

#include "builder.h"
#include "idm.h"

namespace iplf {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// int64_t EvalIO::evalTotalWL(const std::vector<eval::WLNet*>& net_list, const std::string& wl_type)
// {
//   eval::WirelengthEval eval;
//   eval.set_net_list(net_list);
//   return eval.evalTotalWL("kHPWL");
// }

// /// timing eval
// void EvalIO::estimateDelay(std::vector<eval::TimingNet*> timing_net_list, const char* sta_workspace_path, const char* sdc_file_path,
//                            std::vector<const char*> lib_file_path_list)
// {
//   idb::IdbBuilder* idb_builder = dmInst->get_idb_builder();
//   eval::TimingEval eval;
//   eval.set_timing_net_list(timing_net_list);
//   eval.estimateDelay(idb_builder, sta_workspace_path, lib_file_path_list, sdc_file_path);
// }

}  // namespace iplf

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
/**
 * @file ipnp_api.hh
 * @author Jianrong Su
 * @brief
 * @version 1.0
 * @date 2025-06-23
 */

#include "ipnp_api.hh"

#include "PNP.hh"
#include "log/Log.hh"

namespace ipnp {

PNPApi* PNPApi::_instance = nullptr;

void PNPApi::run_pnp(std::string config)
{
  auto pnp = PNP(config);
  // run
  pnp.init();
  pnp.runSynthesis();
  pnp.saveToIdb();
}

void PNPApi::connect_M2_M1(std::string config)
{
  auto pnp = PNP(config);
  pnp.init();
  pnp.connect_M2_M1();
  pnp.saveToIdb();
}

}  // namespace ipnp
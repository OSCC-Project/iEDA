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
 * @file FastPlacer.cpp
 * @author Jianrong Su
 * @brief
 * @version 0.1
 * @date 2024-07-15
 */

#include "FastPlacer.hh"
#include "PLAPI.hh"
#include "log/Log.hh"


namespace ipnp {

void FastPlacer::runFastPlacer(idb::IdbBuilder* idb_builder)
{
  std::string pl_json_file = "/home/sujianrong/iEDA/src/operation/iPNP/api/pl_default_config.json";
  
  ipl::PLAPI& plapi = ipl::PLAPI::getInst();

  plapi.initAPI(pl_json_file, idb_builder);

  // // 只运行全局布局和合法化
  // plapi.runGP();
  // plapi.runLG();

  // // 输出HPWL信息
  // plapi.printHPWLInfo();

  // // 如果启用了时序分析，则输出时序信息
  // if (plapi.isSTAStarted()) {
  //   plapi.printTimingInfo();
  // }

  // // 生成报告
  // plapi.reportPLInfo();
  // LOG_INFO << "Log has been writed to dir: ./result/pl/log/";

  // // 写回数据库
  // plapi.writeBackSourceDataBase();

  plapi.runFlow();

  // 释放资源
  plapi.destoryInst();
}


}  // namespace ipnp

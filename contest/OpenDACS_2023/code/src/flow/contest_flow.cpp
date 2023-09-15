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
 * @File Name: contest_flow.cpp
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2023-09-15
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "contest_flow.h"

#include "contest_evaluation.h"
#include "contest_process.h"

namespace ieda_contest {

ContestFlow::~ContestFlow()
{
  if (_data_manager != nullptr) {
    delete _data_manager;
    _data_manager = nullptr;
  }
}

bool ContestFlow::init(std::string guide_input, std::string guide_output)
{
  if (_data_manager == nullptr) {
    _data_manager = new ContestDataManager(guide_input, guide_output);
  }

  return _data_manager->buildData();
}

void ContestFlow::process()
{
  ContestProcess contest_process(_data_manager);

  contest_process.doProcess();
}

bool ContestFlow::save()
{
  if (_data_manager == nullptr)
    return false;

  return _data_manager->saveData();
}

bool ContestFlow::run_flow(std::string guide_input, std::string guide_output)
{
  if (!init(guide_input, guide_output)) {
    return false;
  }

  process();

  return save();
}

bool ContestFlow::run_evaluation(std::string guide_file, std::string report_file)
{
  if (!init(guide_file)) {
    return false;
  }

  ContestEvaluation contest_evaluation(_data_manager);

  return contest_evaluation.doEvaluation(report_file);
}

}  // namespace ieda_contest

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
 * @File Name: contest_evaluation.cpp
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2023-09-15
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "contest_evaluation.h"

#include <iostream>

#include "contest_dm.h"

namespace ieda_contest {

ContestEvaluation::ContestEvaluation(ContestDataManager* data_manager)
{
  _data_manager = data_manager;
}

bool ContestEvaluation::doEvaluation(std::string report_file)
{
  // overlap检查
  if (!overlapCheckPassed()) {
    std::cout << "Overlap check failed!" << std::endl;
    return false;
  }

  // 连通性检查
  if (!connectivityCheckPassed()) {
    std::cout << "Connectivity check failed!" << std::endl;
    return false;
  }

  // overflow检查
  if (!overflowCheckPassed()) {
    std::cout << "Overflow check failed!" << std::endl;
    return false;
  }

  double score = 0;
  // 计算时序分数
  score += calcTimingScore();
  std::cout << "##############################" << std::endl;
  std::cout << "Final score: " << score << std::endl;
  std::cout << "##############################" << std::endl;

  return true;
}

bool ContestEvaluation::overlapCheckPassed()
{
  return true;
}

bool ContestEvaluation::connectivityCheckPassed()
{
  return true;
}

bool ContestEvaluation::overflowCheckPassed()
{
  return true;
}

double ContestEvaluation::calcTimingScore()
{
  return 101;
}

}  // namespace ieda_contest

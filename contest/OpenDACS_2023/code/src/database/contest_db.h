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
 * @File Name: contest_db.h
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2023-09-15
 *
 */
#pragma once

#include <vector>
#include "contest_guide.h"

namespace ieda_contest {
class ContestDB
{
 public:
  ContestDB();
  ~ContestDB();

  /// getter
  std::vector<ContestGuideNet>& get_guide_nets() { return _guide_nets; }

  /// setter
  void clear()
  {
    _guide_nets.clear();
    std::vector<ContestGuideNet>().swap(_guide_nets);
  }

 private:
  std::vector<ContestGuideNet> _guide_nets;
};

}  // namespace ieda_contest

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

#ifndef IPL_LG_METHOD_H
#define IPL_LG_METHOD_H

#include <vector>

#include "config/LegalizerConfig.hh"
#include "database/LGDatabase.hh"

namespace ipl {

class LGMethodInterface
{
 public:
  virtual ~LGMethodInterface() {}

  virtual void initDataRequirement(LGConfig* lg_config, LGDatabase* lg_database) = 0;
  virtual bool isInitialized() = 0;
  virtual bool runLegalization() = 0;

  virtual void specifyTargetInstList(std::vector<LGInstance*>& target_inst_list) = 0;
  virtual bool runIncrLegalization() = 0;
};

}  // namespace ipl

#endif
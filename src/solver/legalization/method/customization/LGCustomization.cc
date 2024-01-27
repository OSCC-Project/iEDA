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

#include "LGCustomization.hh"

#include "LGDatabase.hh"
#include "LGInstance.hh"
#include "LGInterval.hh"

namespace ieda_solver {

LGCustomization::LGCustomization()
{
}

LGCustomization::~LGCustomization()
{
}

void LGCustomization::initDataRequirement(ipl::LGConfig* lg_config, ipl::LGDatabase* lg_database)
{
}

bool LGCustomization::isInitialized()
{
  return true;
}

void LGCustomization::specifyTargetInstList(std::vector<ipl::LGInstance*>& target_inst_list)
{
}

bool LGCustomization::runLegalization()
{
  return true;
}

bool LGCustomization::runIncrLegalization()
{
  return true;
}

bool LGCustomization::runRollback(bool clear_but_not_rollback)
{
  return true;
}

}  // namespace ieda_solver
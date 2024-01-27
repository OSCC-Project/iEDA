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

#pragma once

#include <vector>

namespace ipl {
class LGConfig;
class LGDatabase;
class LGInstance;
}  // namespace ipl

namespace ieda_solver {

class LGMethodInterface
{
 public:
  LGMethodInterface() {}
  virtual ~LGMethodInterface() {}

  virtual void initDataRequirement(ipl::LGConfig* lg_config, ipl::LGDatabase* lg_database) = 0;
  virtual bool isInitialized() = 0;
  virtual bool runLegalization() = 0;

  virtual void specifyTargetInstList(std::vector<ipl::LGInstance*>& target_inst_list) = 0;
  virtual bool runIncrLegalization() = 0;
  virtual bool runRollback(bool clear_but_not_rollback) = 0;

 protected:
  ipl::LGDatabase* _database = nullptr;
  ipl::LGConfig* _config = nullptr;
  std::vector<ipl::LGInstance*> _target_inst_list;
};

}  // namespace ieda_solver

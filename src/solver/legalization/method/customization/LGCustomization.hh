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

#include "LGMethodInterface.hh"

namespace ipl {
class LGInstance;
class LGInterval;
template <class T>
class Rectangle;
}  // namespace ipl

namespace ieda_solver {

class LGCustomization : public LGMethodInterface
{
 public:
  LGCustomization();
  LGCustomization(const LGCustomization&) = delete;
  LGCustomization(LGCustomization&&) = delete;
  ~LGCustomization();

  LGCustomization& operator=(const LGCustomization&) = delete;
  LGCustomization& operator=(LGCustomization&&) = delete;

  void initDataRequirement(ipl::LGConfig* lg_config, ipl::LGDatabase* lg_database) override;
  bool isInitialized() override;
  void specifyTargetInstList(std::vector<ipl::LGInstance*>& target_inst_list) override;
  bool runLegalization() override;
  bool runIncrLegalization() override;
  bool runRollback(bool clear_but_not_rollback) override;

 private:
};

}  // namespace ieda_solver

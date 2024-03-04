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

#ifndef IPL_LEGALIZER_H
#define IPL_LEGALIZER_H

#include "Config.hh"
#include "LGMethodInterface.hh"
#include "config/LegalizerConfig.hh"
#include "database/LGDatabase.hh"

namespace ieda_solver {
class LGMethodCreator;
}
namespace ipl {

#define LegalizerInst (ipl::Legalizer::getInst())

enum class LG_MODE
{
  kNone,
  kComplete,
  kIncremental
};

class Legalizer
{
 public:
  static Legalizer& getInst();
  static void destoryInst();
  void initLegalizer(Config* pl_config, PlacerDB* placer_db);

  void updateInstanceList();
  void updateInstanceList(std::vector<Instance*> inst_list);
  bool updateInstance(Instance* pl_inst);
  bool updateInstance(std::string pl_inst_name);

  LG_MODE get_mode() const { return _mode; }
  bool runLegalize();
  bool runIncrLegalize();
  bool runRollback(bool clear_but_not_rollback);

  bool isInitialized() { return _mode != LG_MODE::kNone; }

 private:
  static Legalizer* _s_lg_instance;
  LG_MODE _mode;

  LGConfig _config;
  LGDatabase _database;
  std::vector<LGInstance*> _target_inst_list;
  ieda_solver::LGMethodInterface* _method;

  Legalizer() = default;
  Legalizer(const Legalizer&) = delete;
  Legalizer(Legalizer&&) = delete;
  ~Legalizer();
  Legalizer& operator=(const Legalizer&) = delete;
  Legalizer& operator=(Legalizer&&) = delete;

  void initLGConfig(Config* pl_config);
  void initLGDatabase(PlacerDB* placer_db);
  void initLGLayout();
  void wrapRowList();
  void wrapRegionList();
  void wrapCellList();
  void initSegmentList();

  bool checkMapping();
  LGInstance* findLGInstance(Instance* pl_inst);
  bool checkInstChanged(Instance* pl_inst, LGInstance* lg_inst);
  void updateInstanceInfo(Instance* pl_inst, LGInstance* lg_inst);
  void updateInstanceMapping(Instance* pl_inst, LGInstance* lg_inst);

  void alignInstanceOrient();

  int64_t calTotalMovement();
  int64_t calMaxMovement();
  void notifyPLMovementInfo();

  void writebackPlacerDB();
};

}  // namespace ipl

#endif
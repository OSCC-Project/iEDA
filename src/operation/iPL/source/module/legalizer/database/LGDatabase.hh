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
/*
 * @Author: Shijian Chen  chenshj@pcl.ac.cn
 * @Date: 2023-02-08 11:03:27
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @LastEditTime: 2023-02-20 11:09:44
 * @FilePath: /irefactor/src/operation/iPL/source/module/legalizer_refactor/database/LGDatabase.hh
 * @Description: LG database
 *
 *
 */
#ifndef IPL_LGDATABASE_H
#define IPL_LGDATABASE_H

#include <map>
#include <string>
#include <vector>

#include "LGInstance.hh"
#include "LGLayout.hh"
#include "PlacerDB.hh"

namespace ipl {
class LGDatabase
{
 public:
  LGDatabase();
  LGDatabase(const LGDatabase&) = delete;
  LGDatabase(LGDatabase&&) = delete;
  ~LGDatabase();

  LGDatabase& operator=(const LGDatabase&) = delete;
  LGDatabase& operator=(LGDatabase&&) = delete;

  // getter.
  PlacerDB* get_placer_db() const { return _placer_db; }
  int32_t get_shift_x() const { return _shift_x; }
  int32_t get_shift_y() const { return _shift_y; }
  LGLayout* get_lg_layout() const { return _lg_layout; }

  const std::vector<LGInstance*>& get_lgInstance_list() const { return _lgInstance_list; }
  const std::map<LGInstance*, Instance*>& get_lgInstance_map() const { return _lgInstance_map; }
  const std::map<Instance*, LGInstance*>& get_instance_map() const { return _instance_map; }

  // setter.
  void set_placer_db(PlacerDB* placer_db) { _placer_db = placer_db; }
  void set_shift_x(int32_t shift_x) { _shift_x = shift_x; }
  void set_shift_y(int32_t shift_y) { _shift_y = shift_y; }
  void set_lg_layout(LGLayout* lg_layout) { _lg_layout = lg_layout; }
  void add_lgInstance(LGInstance* lg_inst) { _lgInstance_list.push_back(lg_inst); }
  void add_lgInst_to_inst(LGInstance* lg_inst, Instance* inst) { _lgInstance_map.emplace(lg_inst, inst); }
  void add_inst_to_lgInst(Instance* inst, LGInstance* lg_inst) { _instance_map.emplace(inst, lg_inst); }

 private:
  PlacerDB* _placer_db;

  int32_t _shift_x;
  int32_t _shift_y;

  LGLayout* _lg_layout;
  std::vector<LGInstance*> _lgInstance_list;
  std::map<LGInstance*, Instance*> _lgInstance_map;
  std::map<Instance*, LGInstance*> _instance_map;

  friend class Legalizer;
};
}  // namespace ipl
#endif
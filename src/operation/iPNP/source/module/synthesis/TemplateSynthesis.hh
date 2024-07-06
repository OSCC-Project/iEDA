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
 * @brief This file may not be necessary. Repeats the functions of TemlateLib.hh and GenPdnTemplate.py.
 */

#pragma once

#include "GridManager.hh"
#include "TemplateLib.hh"

namespace ipnp {
class TemplateSynthesis
{
 public:
  TemplateSynthesis() = default;
  ~TemplateSynthesis() = default;

  void synthesizeTemplate() {}
  GridManager& getTemplate() { return _tcell_map; }  // return a 3D Template block

 private:
  GridManager _tcell_map;
  // GridMap<TCell>& tcell_map = _database.get_tcell_map();
};

}  // namespace ipnp
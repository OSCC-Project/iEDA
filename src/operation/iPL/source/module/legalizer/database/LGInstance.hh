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
 * @Date: 2023-01-31 20:34:09
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @LastEditTime: 2023-03-07 09:57:44
 * @FilePath: /irefactor/src/operation/iPL/source/module/legalizer_refactor/database/LGInstance.hh
 * @Description: Instance data structure
 *
 *
 */
#ifndef IPL_LGINSTANCE_H
#define IPL_LGINSTANCE_H

#include <string>
#include <vector>

#include "LGCell.hh"
#include "data/Orient.hh"
#include "data/Rectangle.hh"

namespace ipl {

class LGRegion;

enum class LGINSTANCE_STATE
{
  kNone,
  kUnPlaced,
  kPlaced,
  kFixed
};

class LGInstance
{
 public:
  LGInstance() = delete;
  explicit LGInstance(std::string name);
  LGInstance(const LGInstance&) = delete;
  LGInstance(LGInstance&&) = delete;
  ~LGInstance();

  LGInstance& operator=(const LGInstance&) = delete;
  LGInstance& operator=(LGInstance&&) = delete;

  // getter
  int32_t get_index() const { return _index; }
  std::string get_name() const { return _name; }
  LGCell* get_master() const { return _master; }
  Rectangle<int32_t> get_shape() const { return _shape; }
  Point<int32_t> get_coordi() const { return _shape.get_lower_left(); }
  Orient get_orient() const { return _orient; }
  LGINSTANCE_STATE get_state() const { return _state; }
  double get_weight() const { return _weight; }
  LGRegion* get_belong_region() const { return _belong_region; }

  // setter
  void set_index(int32_t index) { _index = index; }
  void set_master(LGCell* master) { _master = master; }
  void set_shape(Rectangle<int32_t> shape) { _shape = shape; }
  void set_orient(Orient orient) { _orient = orient; }
  void set_state(LGINSTANCE_STATE state) { _state = state; }
  void set_belong_region(LGRegion* belong_region) { _belong_region = belong_region; }
  void set_weight(double weight) { _weight = weight; }

  // function
  void updateCoordi(int32_t llx, int32_t lly);

 private:
  int32_t _index;
  std::string _name;
  LGCell* _master;
  Rectangle<int32_t> _shape;
  Orient _orient;
  LGINSTANCE_STATE _state;
  LGRegion* _belong_region;

  double _weight;
};

}  // namespace ipl
#endif
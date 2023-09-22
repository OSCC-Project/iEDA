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
 * @Date: 2023-02-03 19:47:46
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @LastEditTime: 2023-02-09 14:51:01
 * @FilePath: /irefactor/src/operation/iPL/source/module/legalizer_refactor/database/LGRegion.hh
 * @Description: LG Region data struture
 *
 *
 */
#ifndef IPL_LGREGION_H
#define IPL_LGREGION_H

#include <string>
#include <vector>

#include "LGInstance.hh"
#include "data/Rectangle.hh"

namespace ipl {

enum class LGREGION_TYPE
{
  kNone,
  kFence,
  kGuide
};

class LGRegion
{
 public:
  LGRegion() = delete;
  explicit LGRegion(std::string name);
  LGRegion(const LGRegion&) = delete;
  LGRegion(LGRegion&&) = delete;
  ~LGRegion();

  LGRegion& operator=(const LGRegion&) = delete;
  LGRegion& operator=(LGRegion&&) = delete;

  // getter
  int32_t get_index() const { return _index; }
  std::string get_name() const { return _name; }
  LGREGION_TYPE get_type() const { return _type; }
  std::vector<Rectangle<int32_t>> get_shape_list() const { return _shape_list; }
  std::vector<LGInstance*> get_inst_list() const { return _inst_list; }

  // setter
  void set_index(int32_t index) { _index = index; }
  void set_type(LGREGION_TYPE type) { _type = type; }
  void add_shape(Rectangle<int32_t> shape) { _shape_list.push_back(shape); }
  void add_inst(LGInstance* inst) { _inst_list.push_back(inst); }

 private:
  int32_t _index;
  std::string _name;
  LGREGION_TYPE _type;
  std::vector<Rectangle<int32_t>> _shape_list;
  std::vector<LGInstance*> _inst_list;
};
}  // namespace ipl

#endif
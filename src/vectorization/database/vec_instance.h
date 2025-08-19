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
/**
 * @project		vectorization
 * @date		29/7/2025
 * @version		0.1
 * @description
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include <map>
#include <string>
#include <vector>
#include <cstdint>

namespace ivec {

struct VecInstance
{
  uint64_t id;
  int cell_id;
  std::string name;
  int x;
  int y;
  int width;
  int height;
  int llx;
  int lly;
  int urx;
  int ury;
  std::string orient;
  std::string status;
};

class VecInstances
{
 public:
  VecInstances() {}
  ~VecInstances() {}

  // getter
  std::map<int, VecInstance>& get_instance_map() { return _instance_map; }
  VecInstance* get_instance(int id);

  // setter
  void addInstance(VecInstance inst);

  // operator

 private:
  std::map<int, VecInstance> _instance_map;
};

}  // namespace ivec

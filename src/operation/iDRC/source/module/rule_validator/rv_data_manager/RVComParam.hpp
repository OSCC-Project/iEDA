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

namespace idrc {

class RVComParam
{
 public:
  RVComParam() = default;
  RVComParam(int32_t box_size, int32_t expand_size)
  {
    _box_size = box_size;
    _expand_size = expand_size;
  }
  ~RVComParam() = default;
  // getter
  int32_t get_box_size() const { return _box_size; }
  int32_t get_expand_size() const { return _expand_size; }
  // setter
  void set_box_size(const int32_t box_size) { _box_size = box_size; }
  void set_expand_size(const int32_t expand_size) { _expand_size = expand_size; }

 private:
  int32_t _box_size = -1;
  int32_t _expand_size = -1;
};

}  // namespace idrc

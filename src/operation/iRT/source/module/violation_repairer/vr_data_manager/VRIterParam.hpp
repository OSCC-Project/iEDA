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

namespace irt {

class VRIterParam
{
 public:
  VRIterParam() = default;
  VRIterParam(int32_t size, int32_t offset, int32_t schedule_interval)
  {
    _size = size;
    _offset = offset;
    _schedule_interval = schedule_interval;
  }
  ~VRIterParam() = default;
  // getter
  int32_t get_size() const { return _size; }
  int32_t get_offset() const { return _offset; }
  int32_t get_schedule_interval() const { return _schedule_interval; }
  // setter
  void set_size(const int32_t size) { _size = size; }
  void set_offset(const int32_t offset) { _offset = offset; }
  void set_schedule_interval(const int32_t schedule_interval) { _schedule_interval = schedule_interval; }

 private:
  int32_t _size = -1;
  int32_t _offset = -1;
  int32_t _schedule_interval = -1;
};

}  // namespace irt

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

#include <stdint.h>

#include <vector>

namespace idb {

struct JXYCoordinate
{
  int32_t x;
  int32_t y;
};

class JsonXY
{
 public:
  JsonXY() = default;
  ~JsonXY() = default;

  // getter
  size_t get_nums() const { return _coords.size(); }
  const std::vector<JXYCoordinate>& get_coords() const { return _coords; }
  const JXYCoordinate back() const { return _coords.back(); }

  // setter
  void add_coord(int32_t x, int y) { _coords.emplace_back(x, y); }
  void add_coord(const JXYCoordinate& c) { _coords.emplace_back(c); }

  // function
  void clear() { _coords.clear(); }

 private:
  std::vector<JXYCoordinate> _coords;
};

}  // namespace idb

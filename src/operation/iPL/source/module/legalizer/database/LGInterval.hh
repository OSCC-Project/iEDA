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
 * @Date: 2023-02-07 21:18:56
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @LastEditTime: 2023-02-20 11:12:18
 * @FilePath: /irefactor/src/operation/iPL/source/module/legalizer_refactor/database/LGInterval.hh
 * @Description: LG interval data structure
 *
 *
 */
#ifndef IPL_LGINTERVAL_H
#define IPL_LGINTERVAL_H

#include <string>

namespace ipl {
class LGRow;

class LGInterval
{
 public:
  LGInterval() = delete;
  LGInterval(std::string name, int32_t min_x, int32_t max_x);
  LGInterval(const LGInterval&) = delete;
  LGInterval(LGInterval&&) = delete;
  ~LGInterval();

  LGInterval& operator=(const LGInterval&) = delete;
  LGInterval& operator=(LGInterval&&) = delete;

  // getter
  int32_t get_index() const { return _index; }
  std::string get_name() const { return _name; }
  LGRow* get_belong_row() const { return _belong_row; }
  int32_t get_min_x() { return _min_x; }
  int32_t get_max_x() { return _max_x; }

  // setter
  void set_index(int32_t index) { _index = index; }
  void set_belong_row(LGRow* row) { _belong_row = row; }
  void set_min_x(int32_t min_x) { _min_x = min_x; }
  void set_max_x(int32_t max_x) { _max_x = max_x; }

  // function
  void reset();

 private:
  int32_t _index;
  std::string _name; /* row_index + segment_index */
  LGRow* _belong_row;

  int32_t _min_x;
  int32_t _max_x;
};
}  // namespace ipl
#endif
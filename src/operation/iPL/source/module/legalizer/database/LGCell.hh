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
 * @Date: 2023-02-01 17:35:20
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @LastEditTime: 2023-02-09 15:45:06
 * @FilePath: /irefactor/src/operation/iPL/source/module/legalizer_refactor/database/LGCell.hh
 * @Description: Cell Master data structure
 *
 *
 */
#ifndef IPL_LGCELL_H
#define IPL_LGCELL_H

#include <string>
#include <vector>

namespace ipl {

enum class LGCELL_TYPE
{
  kNone,
  kMacro,
  kSequence,
  kStdcell
};

class LGCell
{
 public:
  LGCell() = delete;
  explicit LGCell(std::string name);
  LGCell(const LGCell&) = delete;
  LGCell(LGCell&&) = delete;
  ~LGCell();

  LGCell& operator=(const LGCell&) = delete;
  LGCell& operator=(LGCell&&) = delete;

  // getter
  int32_t get_index() const { return _index; }
  std::string get_name() const { return _name; }
  LGCELL_TYPE get_type() const { return _type; }
  int32_t get_width() const { return _width; }
  int32_t get_height() const { return _height; }

  // setter
  void set_index(int32_t index) { _index = index; }
  void set_type(LGCELL_TYPE type) { _type = type; }
  void set_width(int32_t width) { _width = width; }
  void set_height(int32_t height) { _height = height; }

 private:
  int32_t _index;
  std::string _name;
  LGCELL_TYPE _type;
  int32_t _width;
  int32_t _height;
};
}  // namespace ipl
#endif
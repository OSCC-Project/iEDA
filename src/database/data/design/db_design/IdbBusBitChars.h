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
 * @file IdbBusBitChars.h
 * @author pengming (435788362@qq.com)
 * @brief
 * @version 0.1
 * @date 2022-09-13
 */
#pragma once

namespace idb {

/**
 * @brief
 * Bus Bit Characters:
 *  BUSBITCHARS "delimiterPair" ;
 * example:
 *  BUSBITCHARS "[]" ;
 * Used to specifies the pair of characters when DEF names are mapped to or from other dbs.
 */
class IdbBusBitChars
{
 public:
  IdbBusBitChars();

  // getter
  char getLeftDelimiter() const { return _left_delimiter; }
  char getRightDelimiter() const { return _right_delimiter; }

  // setter
  void setLeftDelimiter(char left_delimiter) { _left_delimiter = left_delimiter; }
  void setRightDelimter(char right_delimiter) { _right_delimiter = right_delimiter; }

 private:
  char _left_delimiter = '[';
  char _right_delimiter = ']';
};

}  // namespace idb
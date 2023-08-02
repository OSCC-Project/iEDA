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
 * @Author: your name
 * @Date: 2022-02-16 12:18:23
 * @LastEditTime: 2022-02-16 13:53:49
 * @LastEditors: Please set LastEditors
 * @Description: ��koroFileHeader�鿴���� ��������:
 * https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: /LJK-iEDA/iEDA/src/iFP/MacroPlacer/checker/checker.cpp
 */
#include "Checker.h"

namespace ipl::imp {

bool Checker::checkOverlape(std::vector<FPInst*> macro_list)
{
  bool pass = true;
  // coordinate of left-down and right-up
  int32_t macro1_ldx, macro1_ldy, macro1_rux, macro1_ruy;
  int size = macro_list.size() - 1;
  for (int i = 0; i < size; ++i) {
    macro1_ldx = macro_list[i]->get_x();
    macro1_ldy = macro_list[i]->get_y();
    macro1_rux = macro1_ldx + macro_list[i]->get_width();
    macro1_ruy = macro1_ldy + macro_list[i]->get_height();
    for (size_t j = i + 1; j < macro_list.size(); ++j) {
      if (macro1_ldx < macro_list[j]->get_x() + macro_list[j]->get_width()
          && macro1_ldy < macro_list[j]->get_y() + macro_list[j]->get_height()) {
        if (macro1_ldx > macro_list[j]->get_x() && macro1_ldy > macro_list[j]->get_y()) {
          cout << "the macro" << macro_list[i]->get_index() << " and macro" << macro_list[j]->get_index() << " are overlap;" << endl;
          pass = false;
        }
      }
    }
  }
  return pass;
}

bool Checker::checkOutborder(std::vector<FPInst*> macro_list, Coordinate* ld, Coordinate* ru)
{
  cout << "-------star check out border------" << endl;
  bool pass = true;
  for (FPInst* macro : macro_list) {
    if (macro->get_x() < ld->_x || macro->get_y() < ru->_y) {
      cout << "the macro" << macro->get_index() << " out border: "
           << "ld(" << macro->get_x() << ", " << macro->get_y() << ")" << endl;
      pass = false;
    }
    if ((macro->get_x() + macro->get_width()) > ru->_x || (macro->get_y() + macro->get_height()) > ru->_y) {
      cout << "the macro" << macro->get_index() << " out border: "
           << "ru(" << macro->get_x() + macro->get_width() << ", " << macro->get_y() + macro->get_height() << ")" << endl;
      pass = false;
    }
  }
  return pass;
}

}  // namespace ipl::imp
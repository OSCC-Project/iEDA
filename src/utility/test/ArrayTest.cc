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
#include <iostream>

#include "Array.hh"
#include "gtest/gtest.h"
TEST(ArrayTest, base) {
  ieda::Array<int, 5> ea(5);
  for (int i = 0; i < 3; i++) {
    ea[i] = i + 2;
  }
  for (ieda::Array<int, 1000>::const_iterator it = ea.begin(); it != ea.end();
       ++it) {
    std::cout << *it << std::endl;
  }

  int size = ea.size();
  bool empty = ea.empty();
  std::cout << size << " " << empty << std::endl;
  int &front = ea.front();
  int &back = ea.back();
  const int *begin = ea.begin();
  const int *end = ea.end();

  std::cout << front << " " << back << std::endl;
  std::cout << *begin << " " << *end << std::endl;

  ea.fill(6);
  for (ieda::Array<int, 1000>::const_iterator it = ea.begin(); it != ea.end();
       ++it) {
    std::cout << *it << std::endl;
  }
}
TEST(ArrayTest, operator) {
  ieda::Array<int, 5> ea1(5);
  ieda::Array<int, 5> ea2(5);

  for (int i = 0; i < 5; i++) {
    ea1[i] = i + 2;
    ea2[i] = i + 2;
  }

  for (ieda::Array<int, 1000>::const_iterator it = ea1.begin(); it != ea1.end();
       ++it) {
    std::cout << *it << std::endl;
  }
  bool isEqual = (ea1 == ea2) ? true : false;
  bool nEqual = (ea1 != ea2) ? true : false;
  std::cout << isEqual << std::endl;
  std::cout << nEqual << std::endl;
  int size = ea1.size();
  bool empty = ea1.empty();
  std::cout << size << " " << empty << std::endl;
  ea1.fill(6);
  for (ieda::Array<int, 1000>::const_iterator it = ea1.begin(); it != ea1.end();
       ++it) {
    std::cout << *it << std::endl;
  }
}
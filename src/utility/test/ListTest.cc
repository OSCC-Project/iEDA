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

#include "List.hh"
#include "gtest/gtest.h"
TEST(ListTest, base) {
  ieda::List<int> el;
  el.push_back(1);
  el.push_back(2);
  el.push_back(3);
  el.push_back(4);
  el.push_back(5);
  el.push_back(6);
  int size = el.size();
  int mSize = el.max_size();
  std::cout << size << std::endl;
  std::cout << mSize << std::endl;
  el.resize(0);
  std::cout << el.size() << std::endl;
  if (el.empty()) {
    std::cout << "元素为空" << std::endl;
  }
  el.push_front(7);
}
TEST(ListTest, addElements) {
  ieda::List<int> el;
  el.push_back(1);
  el.push_back(2);
  el.push_back(3);
  el.push_back(4);
  el.push_back(5);
  el.push_back(6);
  el.push_front(7);
  ieda::List<int>::iterator it = el.begin();
  el.insert(it, 8);
  el.insert(el.begin(), 3, 9);
  ieda::List<int> el2(5, 10);
  el.insert(el.begin(), el2.begin(), ++el2.begin());
  for (it = el.begin(); it != el.end(); it++) {
    std::cout << *it << std::endl;
  }
}
TEST(ListTest, deleteElements) {
  ieda::List<int> el;
  el.push_back(1);
  el.push_back(2);
  el.push_back(3);
  el.push_back(4);
  el.push_back(5);
  el.push_back(6);
  el.pop_front();
  el.pop_back();
  ieda::List<int>::iterator it = el.begin();
  el.erase(it);
  el.erase(el.begin(), ++el.begin());

  for (it = el.begin(); it != el.end(); it++) {
    std::cout << *it << std::endl;
  }
}
TEST(ListTest, getElements) {
  ieda::List<int> el;
  el.push_back(1);
  el.push_back(2);
  el.push_back(3);
  el.push_back(4);
  el.push_back(5);
  el.push_back(6);

  std::cout << el.front() << std::endl;
  std::cout << el.back() << std::endl;
}
TEST(ListTest, others) {
  ieda::List<int> el;
  ieda::List<int> el2;
  el.push_back(1);
  el.push_back(2);
  el.push_back(3);
  el.push_back(4);
  el.push_back(5);
  el.push_back(6);
  el2.assign(6, 7);

  std::cout << el.front() << std::endl;
  std::cout << el.back() << std::endl;
  swap(el, el2);

  for (ieda::List<int>::iterator it = el.begin(); it != el.end(); it++) {
    std::cout << *it << " ";
  }
  std::cout << std ::endl;
  el.splice(el.end(), el2);

  for (ieda::List<int>::iterator it1 = el.begin(); it1 != el.end(); it1++) {
    std::cout << *it1 << " ";
  }
  std::cout << std ::endl;
  ieda::List<int> el3(3, 5);
  ieda::List<int> el4(3, 7);
  ieda::List<int> el5 = el3 + el4;
  el5 += 8;
  // el3.unique();
  for (ieda::List<int>::iterator it2 = el5.begin(); it2 != el5.end(); it2++) {
    std::cout << *it2 << " ";
  }
}
TEST(ListTest, iterator) {
  ieda::List<int> el;
  el.push_back(1);
  el.push_back(2);
  el.push_back(3);
  std::cout << *(el.begin()) << std::endl;
  std::cout << *(--el.end()) << std::endl;
  std::cout << *(el.cbegin()) << std::endl;
  std::cout << *(--el.cend()) << std::endl;
  std::cout << *(el.rbegin()) << std::endl;
  std::cout << *(--el.rend()) << std::endl;
}
TEST(ListTest, operator) {
  ieda::List<int> el1;
  ieda::List<int> el2;
  el1.push_back(1);
  el1.push_back(2);
  el1.push_back(3);
  el2.push_back(4);
  el2.push_back(5);
  el2.push_back(6);

  bool isEqual = (el1 == el2) ? true : false;
  bool nEqual = (el1 != el2) ? true : false;
  bool compare1 = (el1 < el2) ? true : false;
  bool compare2 = (el1 > el2) ? true : false;
  std::cout << isEqual << std::endl;
  std::cout << nEqual << std::endl;
  std::cout << compare1 << std::endl;
  std::cout << compare2 << std::endl;
}

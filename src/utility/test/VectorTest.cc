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

#include "Vector.hh"
#include "gtest/gtest.h"

TEST(VectorTest, Push) {
  ieda::Vector<int> ev;
  ev.push_back(1);
  ev.push_back(2);
  ev.push_back(3);
  for (ieda::Vector<int>::const_iterator it = ev.begin(); it != ev.end();
       ++it) {
    std::cout << *it << std::endl;
  }

  int *data = ev.data();
  const int *data1 = ev.data();
  int &ele = ev.at(2);
  const int &ele1 = ev.at(2);
  int &front = ev.front();
  const int &front1 = ev.front();
  int &back = ev.back();
  const int &back1 = ev.back();

  std::cout << *data << std::endl;
  std::cout << *data1 << std::endl;
  std::cout << ele << std::endl;
  std::cout << ele1 << std::endl;
  std::cout << front << std::endl;
  std::cout << front1 << std::endl;
  std::cout << back << std::endl;
  std::cout << back1 << std::endl;
}

TEST(VectorTest, Pop) {
  ieda::Vector<int> ev;
  ev.push_back(1);
  ev.push_back(2);
  ev.push_back(3);
  ev.pop_back();

  for (ieda::Vector<int>::const_iterator it = ev.begin(); it != ev.end();
       ++it) {
    std::cout << *it << std::endl;
  }
  std::cout << "the size = " << ev.size() << std::endl;
  ev.resize(100);
  std::cout << "after resize,the size = " << ev.size() << std::endl;
  ev.shrink_to_fit();
  std::cout << "after resize,the size = " << ev.size() << std::endl;
}
TEST(VectorTest, Empty) {
  ieda::Vector<int> ev;
  for (int i = 0; i < 256; ++i) {
    ev.push_back(i + 2);
  }
  bool flag = ev.empty();
  int size = ev.size();
  int max_size = ev.max_size();
  int capacity = ev.capacity();
  std::cout << flag << std::endl;
  std::cout << "max size = " << max_size << std::endl;
  std::cout << "size = " << size << std::endl;
  std::cout << "capacity = " << capacity << std::endl;
}
TEST(VectorTest, operator) {
  ieda::Vector<int> ev1;
  ieda::Vector<int> ev2;
  ieda::Vector<int> ev3;

  ev1.push_back(1);
  ev1.push_back(2);
  ev1.push_back(3);
  ev2.push_back(5);
  ev2.push_back(6);
  ev2.push_back(7);
  int val = ev2[2];

  bool com1 = ev1 < ev2;
  bool com2 = ev1 > ev2;
  std::cout << com1 << com2 << std::endl;

  ev3 = ev1 + ev2;
  for (ieda::Vector<int>::const_iterator it3 = ev3.begin(); it3 != ev3.end();
       ++it3) {
    std::cout << *it3 << std::endl;
  }
  ev1 += 4;
  for (ieda::Vector<int>::const_iterator it1 = ev1.begin(); it1 != ev1.end();
       ++it1) {
    std::cout << *it1 << std::endl;
  }
  ieda::Vector<int> ev4;
  ev4.push_back(1);
  ev4.push_back(2);
  ev4.push_back(3);
  bool equ = (ev1 == ev4) ? true : false;
  std::cout << equ << std::endl;
  ev1.pop_back();
  ev1.swap(ev2);
  for (ieda::Vector<int>::const_iterator it3 = ev1.begin(); it3 != ev1.end();
       ++it3) {
    std::cout << *it3 << std::endl;
  }
}
TEST(VectorTest, index) {
  ieda::Vector<int> ev;
  for (int i = 0; i < 222; ++i) {
    ev.push_back(i + 2);
  }
  ev.push_back(1);
  ev.push_back(2);
  ev.push_back(3);

  int count = ev.count(2);
  std::cout << count << std::endl;
  int index = ev.indexOf(5, 0);
  int index1 = ev.endIndexOf(5, 19);
  std::cout << index << std::endl;
  std::cout << index1 << std::endl;
}
TEST(VectorTest, index1) {
  ieda::Vector<int> ev;
  for (int i = 0; i < 222; ++i) {
    ev.push_back(i + 2);
  }
  ev.push_back(1);
  ev.push_back(2);
  ev.push_back(3);

  ieda::Vector<int> ret = ev.mid(3, 8);
  for (ieda::Vector<int>::const_iterator it4 = ret.begin(); it4 != ret.end();
       ++it4) {
    std::cout << *it4 << std::endl;
  }
}

TEST(VectorTest, maxNum) {
  ieda::Vector<int> ev;
  for (int i = 0; i < 280; ++i) {
    ev.push_back(i + 2);
  }

  ieda::Vector<int> ret = ev.mid(270, 8);
  for (ieda::Vector<int>::const_iterator it4 = ret.begin(); it4 != ret.end();
       ++it4) {
    std::cout << *it4 << std::endl;
  }
  ieda::Vector<int> ev1;
  for (int i = 0; i < 5; ++i) {
    ev1.push_back(i + 2);
  }
  for (ieda::Vector<int>::reverse_iterator it5 = ev1.rbegin();
       it5 != ev1.rend(); ++it5) {
    std::cout << *it5 << std::endl;
  }
  ev1.assign(5, 666);
  ev1.emplace(ev1.begin(), 555);
  ev1.erase(ev1.begin() + 1);
  for (ieda::Vector<int>::const_reverse_iterator it5 = ev1.crbegin();
       it5 != ev1.crend(); ++it5) {
    std::cout << *it5 << std::endl;
  }
}

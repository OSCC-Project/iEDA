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
#include "ArrayList.hh"
#include "gtest/gtest.h"

using namespace std;
class Person {
 public:
  Person(){};
  Person(string name, int age) {
    this->m_Name = name;
    this->m_Age = age;
  }
  string m_Name;
  int m_Age;
};

TEST(ArrayListTest, init) {
  ieda::ArrayList<int> a1;
  for (int i = 0; i < 5; i++) {
    int* v = new int(i);
    a1.add(v);
  }
  cout << a1.size() << endl;
  cout << a1.capacity() << endl;

  ieda::Array<Person, 2> arr1(2);
  ieda::Array<Person, 2> arr2(2);
  ieda::Array<Person, 2> arr3(2);
  ieda::Array<Person, 2> arr4(2);
  ieda::Array<Person, 2> arr5(2);
  Person p1("ab", 100);
  Person p2("cd", 30);
  Person p3("dg", 20);
  Person p4("dd", 10);
  Person p5("ghh", 105);
  arr1[0] = p1;
  arr2[0] = p2;
  arr3[0] = p3;
  arr4[0] = p4;
  arr5[0] = p5;
  ieda::ArrayList<ieda::Array<Person, 2>> a2;
  a2.add(&arr1);
  a2.add(&arr2);
  a2.add(&arr3);
  a2.add(&arr4);
  a2.add(&arr5);
  a2.add(&arr5);
  a2.add(&arr5);

  cout << a2.size() << endl;
  cout << a2.capacity() << endl;
}
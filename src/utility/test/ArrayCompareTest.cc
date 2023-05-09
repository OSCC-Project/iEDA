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
//#include <array>
#include <ctime>
#include <iostream>
//#include <vector>
#include "Array.hh"
#include "absl/time/clock.h"
#include "absl/time/time.h"
//#include "Vector.h"
#include "gtest/gtest.h"
TEST(arrayCompare, add) {
  absl::Time startTime, endTime;
  startTime = absl::Now();

  constexpr size_t array_size = 99000000;
  ieda::Array<size_t, 9900000>* ar1 =
      new ieda::Array<size_t, 9900000>(array_size);
  for (size_t i = 0; i < array_size; i++) {
    (*ar1)[i] = i + 1;
  }
  endTime = absl::Now();
  absl::Duration duration = endTime - startTime;
  int64_t duration_time = duration / absl::Nanoseconds(1);
  std::cout << "the array run time is = " << duration_time << "ns" << std::endl;
}
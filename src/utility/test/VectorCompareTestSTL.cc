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
#include <ctime>
#include <iostream>
#include <vector>
//#include "Vector.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gtest/gtest.h"

TEST(vectorCompare, addSTL) {
  absl::Time startTime3, endTime3;
  std::vector<int> ar4;
  startTime3 = absl::Now();
  for (int i = 0; i < 9900000; i++) {
    ar4.push_back(i + 2);
  }
  endTime3 = absl::Now();
  absl::Duration duration = endTime3 - startTime3;
  std::cout << "the STL vector run time is = "
            << duration / absl::Nanoseconds(1) << "ns" << std::endl;
}

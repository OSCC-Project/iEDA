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
 * @Author: S.J Chen
 * @Date: 2022-01-20 12:04:30
 * @LastEditTime: 2022-01-20 17:34:42
 * @LastEditors: S.J Chen
 * @Description:
 * @FilePath: /iEDA/src/iPL/test/GlogTest.cc
 * Contact : https://github.com/sjchanson
 */

#include "gtest/gtest.h"
#include "module/logger/Log.hh"

namespace ipl {

class GlogTest : public testing::Test
{
  void SetUp()
  {
    char  config[] = "glog_test";
    char* argv[]   = {config};
    Log::init(argv);
  }
  void TearDown() { Log::end(); }
};

TEST_F(GlogTest, sample)
{
  LOG_INFO << "This is INFO";
  LOG_WARNING << "This is WARING";
  LOG_ERROR << "This is ERROR";
  //   LOG_FATAL << "This is FATAL";

  std::cout << std::endl;

  DLOG_INFO << "This is INFO only display in DEBUG mode";
  DLOG_WARNING << "This is WARNING only display in DEBUG mode";
  DLOG_ERROR << "This is ERROR only display in DEBUG mode";
  //   DLOG_FATAL << "This is FATAL only display in DEBUG mode";

  std::cout << std::endl;

  bool flag = true;
  LOG_INFO_IF(flag) << "Print the INFO if the condition is true";

  std::cout << std::endl;

  int max_iter = 20;
  for (int i = 0; i < max_iter; i++) {
    LOG_INFO_EVERY_N(5) << "Print the INFO every 5 iteration. "
                        << "Current Iter : " << i;
  }

  std::cout << std::endl;
}

}  // namespace ipl
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
#include "Reporter.h"

namespace ino {

/**
 * @brief report start or end time
 *
 * @param begin
 * true: start time.   false: end time.
 */
void Reporter::reportTime(bool begin) {
  _outfile.open(_output_path, std::ios::app);
  time_t timep;
  time(&timep);
  char tmp[256];
  strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S", localtime(&timep));

  if (begin) {
    _outfile << "\n\n======================= Program start time " << tmp
             << "=======================" << std::endl;
  } else {
    _outfile << "======================= Program end time " << tmp
             << "=======================" << std::endl;
  }
  _outfile.close();
}

void Reporter::report(string info) {
  _outfile.open(_output_path, std::ios::app);
  _outfile << info << std::endl;
  _outfile.close();
}
} // namespace ino

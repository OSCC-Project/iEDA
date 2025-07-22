// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of
// Sciences Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan
// PSL v2. You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file density_io.cpp
 * @author qiming chu (me@emin.chat)
 * @brief Density evaluation IO for tcl interface.
 * @version 0.1
 * @date 2025-06-14
 */

#ifndef DENSITY_IO_H
#define DENSITY_IO_H

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace ieval {
class EvalDensity
{
 public:
  static bool runDensityEvalAndOutput(int grid_size, const std::string& stage);

  static void setOutputPath(const std::string& path);

 private:
  EvalDensity();
  ~EvalDensity();
  std::string _db_config_path;
  static std::string _output_path;
};
}  // namespace ieval
#endif  // DENSITY_IO_H

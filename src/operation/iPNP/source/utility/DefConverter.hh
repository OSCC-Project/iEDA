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
/**
 * @file DefContentExtractor.hh
 * @brief 用于从DEF文件中提取特定内容并插入到另一个文件中
 */

#pragma once

#include <string>
#include <vector>
#include <regex>
#include <fstream>
#include <iostream>

namespace ipnp {

class DefConverter
{
public:
  DefConverter() = default;
  ~DefConverter() = default;

  /**
   * @brief 从源文件提取内容替换目标文件中的相应部分
   * 
   * 该函数会保留目标文件中的起始行和结束行，但替换它们之间的内容。
   * 替换的内容来自源文件中的相应部分（不包括源文件的起始行和结束行）。
   * 
   * @param source_def_path 源DEF文件路径
   * @param target_file_path 目标文件路径
   * @param start_pattern 提取和替换的起始行标记(完全匹配)
   * @param end_pattern 提取和替换的结束行标记(完全匹配)
   */
  void runDefConverter(
    const std::string& source_def_path,
    const std::string& target_file_path,
    const std::string& start_pattern,
    const std::string& end_pattern
  );

private:
  
};

} // namespace ipnp 
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
 * @file DefContentExtractor.cpp
 * @brief 用于从DEF文件中提取特定内容并插入到另一个文件中
 */

#include "DefConverter.hh"

namespace ipnp {

void DefConverter::runDefConverter(
  const std::string& source_def_path,
  const std::string& target_file_path,
  const std::string& start_pattern,
  const std::string& end_pattern
) {
  std::cout << "Running DEF converter..." << std::endl;
  std::cout << "Source file: " << source_def_path << std::endl;
  std::cout << "Target file: " << target_file_path << std::endl;
  std::cout << "Start pattern: " << start_pattern << std::endl;
  std::cout << "End pattern: " << end_pattern << std::endl;
  
  // 1. 读取源文件
  std::ifstream source_file(source_def_path);
  if (!source_file.is_open()) {
    std::cerr << "Error: Cannot open source file: " << source_def_path << std::endl;
    return;
  }

  std::string source_content((std::istreambuf_iterator<char>(source_file)),
                            std::istreambuf_iterator<char>());
  source_file.close();

  // 2. 从源文件中提取内容（不包括起始行和结束行）
  std::string start_line_pattern = "^" + start_pattern + "$";
  std::string end_line_pattern = "^" + end_pattern + "$";
  
  // 使用正则表达式查找起始行和结束行
  std::regex start_regex(start_pattern);
  std::regex end_regex(end_pattern);
  
  std::smatch start_match;
  bool found_start = std::regex_search(source_content, start_match, start_regex);
  if (!found_start) {
    std::cerr << "Error: Start pattern not found in source file" << std::endl;
    return;
  }
  
  // 从起始位置之后查找结束位置
  size_t content_start_pos = start_match.position() + start_match.length();
  std::string remaining = source_content.substr(content_start_pos);
  
  std::smatch end_match;
  bool found_end = std::regex_search(remaining, end_match, end_regex);
  if (!found_end) {
    std::cerr << "Error: End pattern not found in source file" << std::endl;
    return;
  }
  
  // 提取起始行和结束行之间的内容（不包括它们自己）
  std::string extracted_content = remaining.substr(0, end_match.position());
  
  // 3. 读取目标文件
  std::ifstream target_file(target_file_path);
  if (!target_file.is_open()) {
    std::cerr << "Error: Cannot open target file: " << target_file_path << std::endl;
    return;
  }
  
  std::string target_content((std::istreambuf_iterator<char>(target_file)),
                            std::istreambuf_iterator<char>());
  target_file.close();
  
  // 4. 在目标文件中找到起始行和结束行
  found_start = std::regex_search(target_content, start_match, start_regex);
  if (!found_start) {
    std::cerr << "Error: Start pattern not found in target file" << std::endl;
    return;
  }
  
  content_start_pos = start_match.position() + start_match.length();
  remaining = target_content.substr(content_start_pos);
  
  found_end = std::regex_search(remaining, end_match, end_regex);
  if (!found_end) {
    std::cerr << "Error: End pattern not found in target file" << std::endl;
    return;
  }
  
  // 5. 替换目标文件中起始行和结束行之间的内容
  std::string new_content = target_content.substr(0, content_start_pos);
  new_content += extracted_content;
  new_content += remaining.substr(end_match.position());
  
  // 6. 写回到目标文件
  std::ofstream out_file(target_file_path);
  if (!out_file.is_open()) {
    std::cerr << "Error: Cannot write to target file: " << target_file_path << std::endl;
    return;
  }
  
  out_file << new_content;
  out_file.close();
  
  std::cout << "Successfully replaced content in target file" << std::endl;
}

} // namespace ipnp 
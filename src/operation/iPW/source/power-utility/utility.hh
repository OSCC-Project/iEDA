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
#pragma once

#include <deque>
#include <string>

inline std::deque<std::string> split_scope(const std::string& target_scope) {
  auto scope = target_scope;
  scope.erase(std::remove(scope.begin(), scope.end(), '\\'), scope.end());
  // split scope by delimiter "/"
  if (scope.front() == '/') {
    scope.erase(0, 1);
  }
  std::deque<std::string> result;
  std::string temp = "";
  for (size_t i = 0; i < scope.size(); ++i) {
    if (scope[i] == '/') {
      result.push_back(temp);
      temp = "";
    } else {
      temp += scope[i];
    }
  }
  result.push_back(temp);
  return result;
}
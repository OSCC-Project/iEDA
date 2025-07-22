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
 * @file JsonParser.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "CtsConfig.hh"

namespace icts {

using std::string;
using std::vector;

class JsonParser {
 public:
  static JsonParser &getInstance();

  void parse(const string &json_file, CtsConfig *config) const;

 private:
  JsonParser();
  JsonParser(const JsonParser &parser) = delete;
  std::string resolvePath(const std::string& path) const;
  JsonParser &operator=(const JsonParser &) = default;
};
}  // namespace icts
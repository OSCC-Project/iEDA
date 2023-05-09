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

#include "NoConfig.h"
#include "json.hpp"
#include <mutex>

namespace ino {
using Json = nlohmann::json;

class JsonParser {
 public:
  static JsonParser *get_json_parser();

  void parse(const string &json_file, NoConfig *config) const;

 private:
  JsonParser() = default;
  JsonParser(const JsonParser &parser) = delete;
  JsonParser &operator=(const JsonParser &) = default;

  void jsonToConfig(Json *json, NoConfig *config) const;

  void printConfig(NoConfig *config) const;

  static JsonParser *_json_parser;
};
} // namespace ino
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
/**
 * @project		iplf
 * @file		file_drc.h
 * @date		29/3/2023
 * @version		0.1
 * @description


        Process file
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <fstream>
#include <vector>

#include "file_manager.h"
#include "ids.hpp"
#include "json.hpp"

using std::string;
using std::vector;

namespace iplf {

using json = nlohmann::ordered_json;

class FileDrcManager : public FileManager
{
 public:
  explicit FileDrcManager(string data_path) : FileManager(data_path) {}

  explicit FileDrcManager(string data_path, int32_t object_id) : FileManager(data_path, FileModuleId::kDRC, object_id) {}
  ~FileDrcManager() = default;

  virtual bool readFile() override;
  virtual bool saveFileData() override;

 private:
  bool saveJson();
  bool readJson();

  /// pa data

 private:
  constexpr static size_t max_num = 100000;

  void parseJson(std::string path = "");
};

}  // namespace iplf

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
 * @file		file_cts.h
 * @date		25/05/2021
 * @version		0.1
 * @description


        Process file
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "file_manager.h"

namespace iplf {
using namespace std;
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// CTS routing data structure
struct FileInstanceHeader
{
  int32_t instance_num;
  int32_t index;
};

struct FileInstance
{
  //   char instance_name[1000];
  int32_t x;
  int32_t y;
  int8_t orient;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class FilePlacementManager : public FileManager
{
 public:
  explicit FilePlacementManager(string data_path) : FileManager(data_path) {}

  explicit FilePlacementManager(string data_path, int32_t object_id) : FileManager(data_path, FileModuleId::kPL, object_id) {}
  ~FilePlacementManager() = default;

 private:
  /// file parser
  virtual bool parseFileData() override;

  /// file save
  virtual int32_t getBufferSize() override;
  virtual bool saveFileData() override;

  /// pa data

 private:
  bool saveCtsRoutingResult();
  bool parseCtsRoutingResult();
};

}  // namespace iplf

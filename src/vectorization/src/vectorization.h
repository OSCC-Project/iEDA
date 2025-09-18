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
#include <string>

#include "vec_dm.h"
#include "vec_net.h"

namespace ivec {

class Vectorization
{
 public:
  Vectorization();
  ~Vectorization() {}

  bool buildLayoutData(const std::string path = "");
  bool buildGraphData(const std::string path, bool is_placement_mode = false);
  bool buildGraphDataWithoutSave(const std::string path, bool is_placement_mode = false);
  std::map<int, VecNet> getGraph(std::string path);
  void buildFeature(const std::string dir, int patch_row_step, int patch_col_step, bool batch_mode, bool is_placement_mode = false, int sta_mode = 0);
  bool buildPatchData(const std::string dir, bool is_placement_mode = false);
  bool buildPatchData(const std::string dir, int patch_row_step, int patch_col_step, bool is_placement_mode = false);

  bool runVecSTA(const std::string dir);
  bool readNetsToIDB(const std::string dir);
  bool readNetsPatternToIDB(const std::string path);

 private:
  VecDataManager _data_manager;  /// top module data manager

  void initLog(std::string log_path = "");

  void generateFeature(const std::string dir, bool is_placement_mode = false, int sta_mode = 0);
};

}  // namespace ivec
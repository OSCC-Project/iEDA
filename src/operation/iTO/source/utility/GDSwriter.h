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

#include <fstream>
#include <iostream>
#include <string>

namespace idb {
class IdbDesign;
class IdbBlockage;
class IdbInstance;
class IdbBuilder;
}  // namespace idb

namespace ito {
using std::cout;
using std::endl;
using std::ofstream;

class GDSwriter
{
 public:
  GDSwriter() = default;
  ~GDSwriter() = default;

  static void writeGDS(idb::IdbBuilder* idb_builder, std::string path);

  static void writeBlockage(ofstream& gds_file, idb::IdbBlockage* block, int layer, int named);

  static void writeAllInstance(ofstream& gds_file, idb::IdbDesign* idb_design);

  static void writeInstance(ofstream& gds_file, idb::IdbInstance* instance, int layer);
};
}  // namespace ito

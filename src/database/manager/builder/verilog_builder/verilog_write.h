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
 * @file verilog_writer.h
 * @author longshy (longshy@pcl.ac.cn)
 * @brief
 * @version 0.1
 * @date 2021-12-03
 */
#pragma once

#include <stdio.h>

#include <set>
#include <string>
#include <vector>

#include "IdbDesign.h"
#include "IdbEnum.h"
#include "def_service.h"

namespace idb {

class IdbInstance;

class VerilogWriter
{
 public:
  VerilogWriter(const char* file_name, std::set<std::string>& exclude_cell_names, IdbDesign& idb_design, bool is_add_space_for_escape_name);
  ~VerilogWriter();

  void writeModule();
  bool isNeedEscape(const std::string& name);
  std::string escapeName(const std::string& name);
  std::string addSpaceForEscapeName(const std::string& name);
  bool isMiddleSquareBracket(const std::string& str);

 protected:
  void writePorts();
  void writePortDcls();
  void writeWire();
  void writeAssign();
  void writeInstances();
  void writeInstance(IdbInstance* inst);

 private:
  const char* _file_name;
  std::set<std::string> _exclude_cell_names;

  FILE* _stream;
  IdbDesign& _idb_design;
  bool _is_add_space_for_escape_name;
};
}  // namespace idb

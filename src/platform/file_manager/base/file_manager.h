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
 * @file		file_manager.h
 * @date		25/05/2021
 * @version		0.1
 * @description


        Process file
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <fstream>
#include <string>

#include "FileHeader.h"

using std::fstream;
using std::string;

namespace iplf {

class FileManager
{
 public:
  explicit FileManager(string data_path) { _data_path = data_path; }

  explicit FileManager(string data_path, FileModuleId module_id, int32_t object_id)
  {
    _data_path = data_path;
    _file_header._module_id = (int32_t) (module_id);
    _file_header._object_id = object_id;
  }
  virtual ~FileManager() = default;

  // getter
  string get_data_path() { return _data_path; }
  uint64_t get_data_size() { return _file_header._data_size; }
  int32_t get_object_id() { return _file_header._object_id; }

  /// setter
  void set_data_path(string data_path) { _data_path = data_path; }
  void set_module_id(FileModuleId module_id) { _file_header._module_id = (int32_t) (module_id); }
  void set_object_id(int32_t object_id) { _file_header._object_id = object_id; }
  void set_data_size(uint64_t data_size) { _file_header._data_size = data_size; }

  /// file operator
  virtual bool readFile();
  bool writeFile();

  fstream& get_fstream() { return _fstream; }
  bool openFile();
  bool createFile();
  bool closeFile();
  /// file parser
  bool parseFile();
  /// file save
  bool saveFile() { return saveFileHeader() & saveFileData(); }

  /// PA
  bool is_module_pa() { return compareModuleId(FileModuleId::kPA); }
  bool is_pa_data() { return compareObjectId((int32_t) (PaDbId::kPaData)); }

 private:
  string _data_path;
  FileHeader _file_header;
  fstream _fstream;

  /// file parser
  bool parseFileHeader();
  virtual bool parseFileData() { return true; }

  /// file save
  bool saveFileHeader();
  virtual int32_t getBufferSize() { return 0; }
  virtual bool saveFileData() { return true; }

  /// Operator
  bool compareModuleId(FileModuleId id) { return _file_header._module_id == (int32_t) (id) ? true : false; }
  /// PA operate
  bool compareObjectId(int32_t id) { return _file_header._object_id == id ? true : false; }
};

}  // namespace iplf

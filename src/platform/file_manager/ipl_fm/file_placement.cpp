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
 * @project		iplf
 * @file		file_cts.h
 * @date		25/05/2021
 * @version		0.1
 * @description


        Process file.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "file_placement.h"

#include <iostream>

#include "ipl_io/ipl_io.h"

namespace iplf {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int32_t FilePlacementManager::getBufferSize()
{
  return sizeof(FileInstanceHeader) + plInst->get_file_inst_list().size() * sizeof(FileInstance);
}

bool FilePlacementManager::parseFileData()
{
  int size = get_data_size();
  if (size == 0) {
    return false;
  }
  /// clear()
  plInst->clearFileInstanceList();

  char* data_buf = new char[size];
  get_fstream().read(data_buf, size);
  get_fstream().seekp(size, ios::cur);
  char* buf_ref = data_buf;

  /// parse cts data header
  FileInstanceHeader inst_data_header;
  std::memcpy(&inst_data_header, buf_ref, sizeof(FileInstanceHeader));
  buf_ref += sizeof(FileInstanceHeader);

  plInst->initFileInstanceSize(inst_data_header.instance_num);
  for (int i = 0; i < inst_data_header.instance_num; ++i) {
    /// parse instance
    FileInstance instance;
    std::memcpy(&instance, buf_ref, sizeof(FileInstance));

    /// add to segment list
    plInst->addFileInstance(instance);
    buf_ref += sizeof(FileInstance);
  }
  delete[] data_buf;
  data_buf = nullptr;
  buf_ref = nullptr;

  return true;
}

bool FilePlacementManager::saveFileData()
{
  int size = getBufferSize();
  //   assert(size != 0);

  char* data_buf = new char[size];
  std::memset(data_buf, 0, size);

  int mem_size = 0;
  char* buf_ref = data_buf;

  /// save cts data header
  FileInstanceHeader instance_data_header;
  instance_data_header.instance_num = plInst->get_file_inst_list().size();
  instance_data_header.index = plInst->get_dp_index();
  std::memcpy(buf_ref, &instance_data_header, sizeof(FileInstanceHeader));
  buf_ref += sizeof(FileInstanceHeader);
  mem_size += sizeof(FileInstanceHeader);

  for (auto& instance : plInst->get_file_inst_list()) {
    /// save instance
    std::memcpy(buf_ref, &instance, sizeof(FileInstance));
    buf_ref += sizeof(FileInstance);
    mem_size += sizeof(FileInstance);
  }

  if (size != mem_size) {
    std::cout << "Error : memory size error." << std::endl;
  }

  /// write file
  get_fstream().write(data_buf, size);

  get_fstream().seekp(size, ios::cur);

  delete[] data_buf;
  data_buf = nullptr;
  buf_ref = nullptr;

  return true;
}

}  // namespace iplf

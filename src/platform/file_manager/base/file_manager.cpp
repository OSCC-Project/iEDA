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
 * @file		file_manager.h
 * @date		25/05/2021
 * @version		0.1
 * @description


        Process file.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "file_manager.h"

#include <cstring>
#include <iostream>

using namespace std;

namespace iplf {
bool FileManager::readFile()
{
  if (openFile()) {
    bool result = parseFile();

    closeFile();

    std::cout << "[FileManager Info] : Read file success. Path = " << _data_path << std::endl;
    return result;
  }

  return false;
}

bool FileManager::writeFile()
{
  if (!createFile()) {
    return false;
  }

  bool result = saveFile();

  closeFile();

  std::cout << "[FileManager Info] : Write file success. Path = " << _data_path << std::endl;

  return result;
}

bool FileManager::openFile()
{
  _fstream.open(_data_path, ios_base::in | ios_base::binary);
  if (!_fstream.is_open()) {
    std::cout << "[FileManager Error] : Open file failed. Path = " << _data_path << std::endl;
    return false;
  }
  std::cout << "[FileManager Info] : Open file success. Path = " << _data_path << std::endl;

  _fstream.seekg(ios::beg);

  return true;
}

bool FileManager::createFile()
{
  _fstream.clear();
  _fstream.open(_data_path, ios_base::in | ios_base::out | ios_base::binary | ios::trunc);
  if (!_fstream.is_open()) {
    std::cout << "[FileManager Error] : Create file failed. Path = " << _data_path << std::endl;
    return false;
  }

  _fstream.seekg(ios::beg);

  std::cout << "[FileManager Info] : Create file success. Path = " << _data_path << std::endl;
  return true;
}

bool FileManager::closeFile()
{
  _fstream.close();
  std::cout << "[FileManager Info] : Close file success. Path = " << _data_path << std::endl;
  return true;
}

bool FileManager::parseFile()
{
  if (parseFileHeader()) {
    return parseFileData();
  }

  return false;
}

bool FileManager::parseFileHeader()
{
  _fstream.read((char*) (&_file_header), sizeof(FileHeader));
  _fstream.seekp(sizeof(FileHeader), ios::cur);

  return _fstream.fail() ? false : true;
}

bool FileManager::saveFileHeader()
{
  _file_header._data_size = getBufferSize();

  _fstream.write((char*) &_file_header, sizeof(FileHeader));

  _fstream.seekp(sizeof(FileHeader), ios::cur);

  return _fstream.fail() ? false : true;
}

}  // namespace iplf

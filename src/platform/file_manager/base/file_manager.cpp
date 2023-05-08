/**
 * iEDA
 * Copyright (C) 2021  PCL
 *
 * This program is free software;
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @project		iplf
 * @file		file_manager.h
 * @copyright	(c) 2021 All Rights Reserved.
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

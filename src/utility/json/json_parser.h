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
#pragma GCC diagnostic ignored "-Wunused-function"
/**
 * @File Name: json_parser.h
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2022-04-15
 *
 */
#include <sys/resource.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include <cassert>
#include <cmath>
#include <ctime>
#include <fstream>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <vector>

#include "../string/Str.hh"
#include "json.hpp"
#include "zlib.h"

namespace ieda {

static nlohmann::json getJsonData(nlohmann::json value, std::vector<std::string> flag_list, nlohmann::json default_value = "")
{
  if (flag_list.empty()) {
    std::cout << "[json error] : The flag list is empty!" << std::endl;
  }

  int flag_size = flag_list.size();
  for (int i = 0; i < flag_size; i++) {
    value = value[flag_list[i]];
  }

  if (!value.is_null()) {
    return value;
  }

  std::string key;

  for (int i = 0; i < flag_size; i++) {
    key += flag_list[i];
    if (i < flag_size - 1) {
      key += ".";
    }
  }
  //   std::cout << "[json error] : The configuration file key = " << key << " do not exist." << std::endl;
  return default_value;
}

template <typename T>
static T& getFileStream(std::string file_path)
{
  T* file = new T(file_path);
  if (!file->is_open()) {
    std::cout << "[json error] : Failed to open file = " << file_path << std::endl;
  }
  return *file;
}

static std::string get_gz_string(std::string file_path)
{
  gzFile file = gzopen(file_path.c_str(), "rb");
  if (!file) {
    std::cout << "[json error] : Failed to open file = " << file_path << std::endl;
    return "";
  }

  unsigned int file_length = 0;
  gzseek(file, 0, SEEK_END);
  file_length = gztell(file);
  gzseek(file, 0, SEEK_SET);

  char* content = (char*) malloc(file_length);
  if (!content) {
    std::cout << "[json warning] : File empty." << file_path << std::endl;
    gzclose(file);
    return "";
  }

  int bytes_read = gzread(file, content, file_length);
  if (bytes_read < 0) {
    printf("Error reading from file\n");
    free(content);
    gzclose(file);
    return std::string(content);
  }

  gzclose(file);

  return std::string(content);
}

static std::istringstream& getGzFileStream(std::string file_path)
{
  if (ieda::Str::contain(file_path.c_str(), ".gz")) {
    std::cout << "[json error] : do not support gz file by now." << std::endl;
    auto content = get_gz_string(file_path);
    std::istringstream* dataStream = new std::istringstream(content);
    return *dataStream;
  } else {
    std::istringstream* dataStream = new std::istringstream("");
    return *dataStream;
  }
}

static std::ifstream& getInputFileStream(std::string file_path)
{
  //   if (ieda::Str::contain(file_path.c_str(), ".gz")) {
  //     std::cout << "[json error] : do not support gz file by now." << std::endl;
  //     auto content = get_gz_string(file_path);
  //     std::istringstream dataStream(content);
  //     std::ifstream* file = new std::ifstream(dataStream);
  //     if (!file->is_open()) {
  //       std::cout << "[json error] : Failed to open file = " << file_path << std::endl;
  //     }
  //     return *file;
  //   } else {
  return getFileStream<std::ifstream>(file_path);
  //   }
}

static void initJson(std::string file_path, nlohmann::json& json)
{
  if (ieda::Str::contain(file_path.c_str(), ".gz")) {
    auto& file_stream = getGzFileStream(file_path);
    file_stream >> json;
  } else {
    auto& file_stream = getFileStream<std::ifstream>(file_path);
    file_stream >> json;
  }
}

static std::ofstream& getOutputFileStream(std::string file_path)
{
  return getFileStream<std::ofstream>(file_path);
}

template <typename T>
static void closeFileStream(T& t)
{
  t.close();
}

template <typename T, typename... Args>
static std::string splice(T value, Args... args)
{
  std::stringstream oss;
  pushStream(oss, value, args...);
  std::string string = oss.str();
  oss.clear();
  return string;
}

template <typename Stream, typename T, typename... Args>
static void pushStream(Stream& stream, T t, Args... args)
{
  stream << t;
  pushStream(stream, args...);
  return;
}

template <typename Stream, typename T>
static void pushStream(Stream& stream, T t)
{
  stream << t;
}

}  // namespace ieda


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

#include "json.hpp"

namespace ieda {

static nlohmann::json getJsonData(nlohmann::json value, std::vector<std::string> flag_list)
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
  std::cout << "[json error] : The configuration file key = " << key << " do not exist." << std::endl;
  return value;
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

static std::ifstream& getInputFileStream(std::string file_path)
{
  return getFileStream<std::ifstream>(file_path);
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

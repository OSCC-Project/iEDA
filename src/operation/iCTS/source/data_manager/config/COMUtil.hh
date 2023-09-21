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
 * @file COMUtil.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#pragma once
#include <sys/resource.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include <cassert>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <map>
#include <queue>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "json/json.hpp"
namespace icts {
class COMUtil {
 public:
  COMUtil() {}

  ~COMUtil() {}

  template <typename T = nlohmann::json>
  static T getData(nlohmann::json value, std::vector<std::string> flag_list) {
    if (flag_list.empty()) {
      return value;
    }
    for (size_t i = 0; i < flag_list.size(); i++) {
      value = value[flag_list[i]];
    }
    if (!value.is_null()) {
      return value;
    }

    std::string key;
    for (size_t i = 0; i < flag_list.size(); i++) {
      key += flag_list[i] + ".";
    }

    return value;
  }

  template <typename T>
  static std::vector<T> getSerializeObjectData(nlohmann::json value,
                                               const std::string& target, T data) {
    std::vector<T> data_list;
    for (size_t i = 0; i < value.size(); ++i) {
      data_list.push_back(COMUtil::getData(value[i], {target}));
    }
    return data_list;
  }

  static std::ifstream &getInputFileStream(const std::string &file_path) {
    return getFileStream<std::ifstream>(file_path);
  }

  static std::ofstream &getOutputFileStream(const std::string &file_path) {
    return getFileStream<std::ofstream>(file_path);
  }

  static double microtime() {
    struct timeval tv;
    struct timezone tz;
    gettimeofday(&tv, &tz);
    return static_cast<double>(tv.tv_sec) +
           static_cast<double>(tv.tv_usec) / 1000000.00;
  }

  static void split(const std::string &s, std::vector<std::string> &v,
                    const std::string &c = " ") {
    v.clear();
    std::string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    while (std::string::npos != pos2) {
      v.push_back(s.substr(pos1, pos2 - pos1));

      pos1 = pos2 + c.size();
      pos2 = s.find(c, pos1);
    }
    if (pos1 != s.length()) {
      v.push_back(s.substr(pos1));
    }
  }

  static void splitWithBlank(const std::string &s,
                             std::vector<std::string> &d) {
    std::string word;
    d.clear();
    std::istringstream record(s);
    while (record >> word) {
      d.push_back(word);
    }
  }

  static void splitStringToDouble(const std::string &s, std::vector<double> &d,
                                  const std::string &c = " ") {
    d.clear();
    std::string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    double dou;
    while (std::string::npos != pos2) {
      std::istringstream is(s.substr(pos1, pos2 - pos1));
      is >> dou;
      d.push_back(dou);
      pos1 = pos2 + c.size();
      pos2 = s.find(c, pos1);
      is.clear();
    }
    if (pos1 != s.length()) {
      std::istringstream is(s.substr(pos1, pos2 - pos1));
      is >> dou;
      d.push_back(dou);
      is.clear();
    }
  }

  static void splitStringToInt(const std::string &s, std::vector<int32_t> &d,
                               const std::string &c = " ") {
    d.clear();
    std::string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    int32_t dou;
    while (std::string::npos != pos2) {
      std::istringstream is(s.substr(pos1, pos2 - pos1));
      is >> dou;
      d.push_back(dou);
      pos1 = pos2 + c.size();
      pos2 = s.find(c, pos1);
      is.clear();
    }
    if (pos1 != s.length()) {
      std::istringstream is(s.substr(pos1, pos2 - pos1));
      is >> dou;
      d.push_back(dou);
      is.clear();
    }
  }

 private:
  template <typename T>
  static T &getFileStream(std::string file_path) {
    T *file = new T(file_path);
    if (!file->is_open()) {
      return *file;
    }
    return *file;
  }
};
}  // namespace icts

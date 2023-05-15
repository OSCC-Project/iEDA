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
 * @File Name: report_manager.h
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2022-11-07
 *
 */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#define rptInst iplf::ReportManager::getInstance()

namespace iplf {

class ReportOStream
{
public:
    explicit ReportOStream(const std::string &file);
    ~ReportOStream();

    bool fileOpen() { return _fs.is_open(); }
    template <typename T>
    ReportOStream &operator<<(T &&obj)
    {
        (_fs.is_open() ? _fs : std::cout) << obj;
        return *this;
    };

private:
    std::fstream _fs;
};


class ReportManager
{
 public:
  static ReportManager* getInstance()
  {
    if (!_instance) {
      _instance = new ReportManager;
    }
    return _instance;
  }

  bool reportDBSummary(const std::string& file_name);
  bool reportWL(const std::string& file_name);
  bool reportCongestion(const std::string& file_name);

  bool reportInstance(const std::string& file_name, const std::string& inst_name);
  bool reportNet(const std::string& file_name, const std::string& net_name);
  bool reportDanglingNet(const std::string& file_name);

  bool reportRoute(const std::string& file, const std::string& net_name, bool summary);
  bool reportPlaceDistribution(const std::vector<std::string>& prefixes, const std::string& file = "");
  bool reportInstLevel(const std::string& prefix, int level, int num_threshold);

  bool reportDRC(const std::string& file);
  bool reportDRC(const std::string& file_name, std::map<std::string, int>& drc_result,
                 std::tuple<bool, std::vector<std::string>, std::vector<std::string>, int>& connectivity_result);

 private:
  static ReportManager* _instance;
  ReportManager() = default;
  ~ReportManager() = default;
};

}  // namespace iplf
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
 * @file iPNPApi.hh
 * @author Xinhao li
 * @brief
 * @version 0.1
 * @date 2024-07-15
 */

#pragma once
#include <iostream>
#include <string>

#include "iPNP.hh"

#define pnpApiInst ipdn::PdnApi::getInstance()
namespace ipnp {

class iPNPApi
{
 public:
  iPNPApi() = default;
  iPNPApi(const std::string& config_file_path);
  ~iPNPApi() = default;

  static iPNPApi* getInstance()
  {
    if (!_instance) {
      _instance = new iPNPApi;
    }
    return _instance;
  }

  static void destroyInst()
  {
    if (_instance != nullptr) {
      delete _instance;
      _instance = nullptr;
    }
  }

  void runiPNP(std::string config_file);

 private:
  std::string _config_file_path;
  static iPNPApi* _instance;
};

}  // namespace ipnp
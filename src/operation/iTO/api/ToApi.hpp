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

#include <any>
#include <map>
#include <string>
#include <vector>

#include "../iTO.h"
#include "define.h"
#include "ids.hpp"

namespace ieda_feature {
class TimingOptSummary;
}  // namespace ieda_feature

namespace ito {

#define ToApiInst (ito::ToApi::getInst())

class ToApi
{
 public:
  static ToApi& getInst()
  {
    if (_instance == nullptr) {
      _instance = new ToApi();
    }

    return *_instance;
  }
  static void destroyInst();

  void init(const std::string& ITO_CONFIG_PATH);
  void initEngine();
  // function API
  void runTO();
  // opt DRV functions
  void optimizeDrv();
  void optimizeDrvSpecialNet(const char* net_name);

  // opt setup functions
  void optimizeSetup();
  void performBuffering(const char* net_name);

  // opt hold functions
  void optimizeHold();

  void saveDef(std::string saved_def_path = "");

  void resetConfigLibs(std::vector<std::string>& paths);
  void resetConfigSdc(std::string& path);

  void reportTiming();

  ieda_feature::TimingOptSummary outputSummary();

 private:
  static ToApi* _instance;
  ToApi();
  ~ToApi();

  ito::iTO* _ito = nullptr;
};

}  // namespace ito

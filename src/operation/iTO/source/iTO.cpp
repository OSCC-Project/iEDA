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
#include "iTO.h"

#include "HoldOptimizer.h"
#include "JsonParser.h"
#include "SetupOptimizer.h"
#include "ToConfig.h"
#include "ViolationOptimizer.h"
#include "data_manager.h"
#include "timing_engine.h"

namespace ito {

iTO::iTO(const std::string& config_file)
{
  JsonParser json(config_file);
  json.parse();
}

void iTO::runTO()
{
  if (toConfig->get_optimize_drv()) {
    optimizeDrv();
  }
  if (toConfig->get_optimize_hold()) {
    optimizeHold();
  }
  if (toConfig->get_optimize_setup()) {
    optimizeSetup();
  }
}

void iTO::optimizeDrv()
{
  cout << "\033[1;32m" << endl;
  cout << R"(    ____        _   _           _           _____  _______      __ )" << endl;
  cout << R"(   / __ \      | | (_)         (_)         |  __ \|  __ \ \    / / )" << endl;
  cout << R"(  | |  | |_ __ | |_ _ _ __ ___  _ _______  | |  | | |__) \ \  / /  )" << endl;
  cout << R"(  | |  | | '_ \| __| | '_ ` _ \| |_  / _ \ | |  | |  _  / \ \/ /   )" << endl;
  cout << R"(  | |__| | |_) | |_| | | | | | | |/ /  __/ | |__| | | \ \  \  /    )" << endl;
  cout << R"(   \____/| .__/ \__|_|_| |_| |_|_/___\___| |_____/|_|  \_\  \/     )" << endl;
  cout << R"(         | |                                                       )" << endl;
  cout << R"(         |_|                                                       )" << endl;
  cout << R"(                                                                   )" << endl;
  cout << "\033[0m" << endl;

  toOptDrv->fixViolations();
}

void iTO::optimizeDrvSpecialNet(const char* net_name)
{
  toOptDrv->fixSpecialNet(net_name);
}

void iTO::optimizeHold()
{
  cout << "\033[1;33m" << endl;
  cout << R"(   ____        _   _           _           _    _       _     _  )" << endl;
  cout << R"(  / __ \      | | (_)         (_)         | |  | |     | |   | | )" << endl;
  cout << R"( | |  | |_ __ | |_ _ _ __ ___  _ _______  | |__| | ___ | | __| | )" << endl;
  cout << R"( | |  | | '_ \| __| | '_ ` _ \| |_  / _ \ |  __  |/ _ \| |/ _` | )" << endl;
  cout << R"( | |__| | |_) | |_| | | | | | | |/ /  __/ | |  | | (_) | | (_| | )" << endl;
  cout << R"(  \____/| .__/ \__|_|_| |_| |_|_/___\___| |_|  |_|\___/|_|\__,_| )" << endl;
  cout << R"(        | |                                                      )" << endl;
  cout << R"(        |_|                                                      )" << endl;
  cout << R"(                                                                 )" << endl;
  cout << "\033[0m" << endl;

  // HoldOptimizer* hold_optimizer = new HoldOptimizer();
  toOptHold->optimizeHold();
}

void iTO::optimizeSetup()
{
  cout << "\033[1;34m" << endl;
  cout << R"(    ____        _   _           _            _____      _                )" << endl;
  cout << R"(   / __ \      | | (_)         (_)          / ____|    | |               )" << endl;
  cout << R"(  | |  | |_ __ | |_ _ _ __ ___  _ _______  | (___   ___| |_ _   _ _ __   )" << endl;
  cout << R"(  | |  | | '_ \| __| | '_ ` _ \| |_  / _ \  \___ \ / _ \ __| | | | '_ \  )" << endl;
  cout << R"(  | |__| | |_) | |_| | | | | | | |/ /  __/  ____) |  __/ |_| |_| | |_) | )" << endl;
  cout << R"(   \____/| .__/ \__|_|_| |_| |_|_/___\___| |_____/ \___|\__|\__,_| .__/  )" << endl;
  cout << R"(         | |                                                     | |     )" << endl;
  cout << R"(         |_|                                                     |_|     )" << endl;
  cout << R"(                                                                         )" << endl;
  cout << "\033[0m" << endl;

  toOptSetup->optimizeSetup();
}

void iTO::performBuffering(const char* net_name) {
  toOptSetup->performBuffering(net_name);
}

}  // namespace ito

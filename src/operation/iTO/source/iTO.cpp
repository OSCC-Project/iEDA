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
#include "DbInterface.h"
#include "HoldOptimizer.h"
#include "JsonParser.h"
#include "SetupOptimizer.h"
#include "ViolationOptimizer.h"
#include "io/DbInterface.h"

namespace ito {

iTO::iTO(const std::string &config_file) {
  _to_config = new ToConfig;
  JsonParser *json = JsonParser::get_json_parser();
  json->parse(config_file, _to_config);
}

iTO::~iTO() {
  _db_interface->destroyDbInterface();
}

void iTO::runTO() {
  if (_to_config->get_optimize_drv()) {
    optimizeDesignViolation();
  }
  if (_to_config->get_optimize_hold()) {
    optimizeHold();
  }
  if (_to_config->get_optimize_setup()) {
    optimizeSetup();
  }
}

void iTO::initialization(idb::IdbBuilder *idb_build, ista::TimingEngine *timing) {
  cout << "\033[1;31m" << endl;
  cout << R"(  _____       _ _     _____        _         )" << endl;
  cout << R"( |_   _|     (_) |   |  __ \      | |        )" << endl;
  cout << R"(   | |  _ __  _| |_  | |  | | __ _| |_ __ _  )" << endl;
  cout << R"(   | | | '_ \| | __| | |  | |/ _` | __/ _` | )" << endl;
  cout << R"(  _| |_| | | | | |_  | |__| | (_| | || (_| | )" << endl;
  cout << R"( |_____|_| |_|_|\__| |_____/ \__,_|\__\__,_| )" << endl;
  cout << R"(                                             )" << endl;
  cout << "\033[0m" << endl;
  _db_interface = ito::DbInterface::get_db_interface(_to_config, idb_build, timing);
};

void iTO::resetInitialization(idb::IdbBuilder    *idb_build,
                              ista::TimingEngine *timing_engine) {
  DbInterface::destroyDbInterface();
  _db_interface =
      ito::DbInterface::get_db_interface(_to_config, idb_build, timing_engine);
}

void iTO::optimizeDesignViolation() {
  cout << "\033[1;32m" << endl;
  cout << R"(    ____        _   _           _           _____  _______      __ )"
       << endl;
  cout << R"(   / __ \      | | (_)         (_)         |  __ \|  __ \ \    / / )"
       << endl;
  cout << R"(  | |  | |_ __ | |_ _ _ __ ___  _ _______  | |  | | |__) \ \  / /  )"
       << endl;
  cout << R"(  | |  | | '_ \| __| | '_ ` _ \| |_  / _ \ | |  | |  _  / \ \/ /   )"
       << endl;
  cout << R"(  | |__| | |_) | |_| | | | | | | |/ /  __/ | |__| | | \ \  \  /    )"
       << endl;
  cout << R"(   \____/| .__/ \__|_|_| |_| |_|_/___\___| |_____/|_|  \_\  \/     )"
       << endl;
  cout << R"(         | |                                                       )"
       << endl;
  cout << R"(         |_|                                                       )"
       << endl;
  cout << R"(                                                                   )"
       << endl;
  cout << "\033[0m" << endl;
  ViolationOptimizer *drv_optimizer = new ViolationOptimizer(_db_interface);
  drv_optimizer->fixViolations();
}

void iTO::optimizeHold() {
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
  HoldOptimizer *hold_optimizer = new HoldOptimizer(_db_interface);
  hold_optimizer->optimizeHold();
  // _hold_optimizer->insertHoldDelay("BUF_X1", "_862_:D");
}

void iTO::optimizeSetup() {
  cout << "\033[1;34m" << endl;
  cout << R"(    ____        _   _           _            _____      _                )"
       << endl;
  cout << R"(   / __ \      | | (_)         (_)          / ____|    | |               )"
       << endl;
  cout << R"(  | |  | |_ __ | |_ _ _ __ ___  _ _______  | (___   ___| |_ _   _ _ __   )"
       << endl;
  cout << R"(  | |  | | '_ \| __| | '_ ` _ \| |_  / _ \  \___ \ / _ \ __| | | | '_ \  )"
       << endl;
  cout << R"(  | |__| | |_) | |_| | | | | | | |/ /  __/  ____) |  __/ |_| |_| | |_) | )"
       << endl;
  cout << R"(   \____/| .__/ \__|_|_| |_| |_|_/___\___| |_____/ \___|\__|\__,_| .__/  )"
       << endl;
  cout << R"(         | |                                                     | |     )"
       << endl;
  cout << R"(         |_|                                                     |_|     )"
       << endl;
  cout << R"(                                                                         )"
       << endl;
  cout << "\033[0m" << endl;
  SetupOptimizer *setup_optimizer = new SetupOptimizer(_db_interface);
  setup_optimizer->optimizeSetup();
}

} // namespace ito

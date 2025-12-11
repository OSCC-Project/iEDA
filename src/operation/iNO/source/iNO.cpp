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
#include "iNO.h"

namespace ino {

iNO::iNO(const std::string &config_file) {
  _no_config = new NoConfig;
  JsonParser *json = JsonParser::get_json_parser();
  json->parse(config_file, _no_config);
//   initialization(config_file, idb_build);
}

iNO::~iNO() {
  if (_db_interface != nullptr) {
    _db_interface->destroyDbInterface();
  }
}

void iNO::initialization(idb::IdbBuilder *idb_build, ista::TimingEngine *timing) {
  cout << "\033[1;31m" << endl;
  cout << R"(  _____       _ _     _____        _         )" << endl;
  cout << R"( |_   _|     (_) |   |  __ \      | |        )" << endl;
  cout << R"(   | |  _ __  _| |_  | |  | | __ _| |_ __ _  )" << endl;
  cout << R"(   | | | '_ \| | __| | |  | |/ _` | __/ _` | )" << endl;
  cout << R"(  _| |_| | | | | |_  | |__| | (_| | || (_| | )" << endl;
  cout << R"( |_____|_| |_|_|\__| |_____/ \__,_|\__\__,_| )" << endl;
  cout << R"(                                             )" << endl;
  cout << "\033[0m" << endl;
  _db_interface = ino::DbInterface::get_db_interface(_no_config, idb_build, timing);
}

void iNO::fixIO() {
  cout << "fixIO" << endl;
  ino::FixFanout *fix_fanout = new FixFanout(_db_interface);
  fix_fanout->fixIO();
  delete fix_fanout;
}

void iNO::fixFanout() {
  cout << "\033[1;35m" << endl;
  cout << R"(  ______ _        ______                      _    )" << endl;
  cout << R"( |  ____(_)      |  ____|                    | |   )" << endl;
  cout << R"( | |__   ___  __ | |__ __ _ _ __   ___  _   _| |_  )" << endl;
  cout << R"( |  __| | \ \/ / |  __/ _` | '_ \ / _ \| | | | __| )" << endl;
  cout << R"( | |    | |>  <  | | | (_| | | | | (_) | |_| | |_  )" << endl;
  cout << R"( |_|    |_/_/\_\ |_|  \__,_|_| |_|\___/ \__,_|\__| )" << endl;
  cout << R"(                                                   )" << endl;
  cout << "\033[0m" << endl;
  ino::FixFanout *fix_fanout = new FixFanout(_db_interface);
  fix_fanout->fixFanout();
  delete fix_fanout;
}

} // namespace ino

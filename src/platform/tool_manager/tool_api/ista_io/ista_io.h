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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

#define staInst iplf::StaIO::getInstance()

namespace ista {
class StaClockTree;
}

namespace idb {
class IdbBuilder;
enum class IdbConnectType : uint8_t;
}  // namespace idb

namespace iplf {

class StaIO
{
 public:
  static StaIO* getInstance()
  {
    if (!_instance) {
      _instance = new StaIO;
    }
    return _instance;
  }

  /// getter

  /// io
  bool autoRunSTA(std::string path = "");
  bool initSTA(std::string path = "", bool init_log = false);
  bool isInitSTA();
  unsigned buildGraph();
  bool runSTA(std::string path = "");
  unsigned updateTiming();
  bool buildClockTree(std::string sta_path = "");

  std::vector<std::unique_ptr<ista::StaClockTree>>& getClockTree();

  /// operator
  std::vector<std::string> getClockNetNameList();
  std::vector<std::string> getClockNameList();
  std::string getCellType(const char* cell_name);
  bool isClockNet(std::string net_name);
  bool isSequentialCell(std::string instance_name);
  bool insertBuffer(std::pair<std::string, std::string>& source_sink_net, std::vector<std::string>& sink_pin_list,
                    std::pair<std::string, std::string>& master_inst_buffer, std::pair<int, int> buffer_center_loc,
                    idb::IdbConnectType connect_type);
  float obtainInstPinCap(std::string inst_pin_name);
  float obtainPinCap(std::string inst_pin_name);
  float obtainAvgWireResUnitLengthUm();
  float obtainAvgWireCapUnitLengthUm();
  float obtainInstOutPinRes(std::string cell_name, std::string port_name);

  bool setStaWorkDirectory(std::string path = "");
  bool readIdb(idb::IdbBuilder* idb_builder = nullptr);
  bool runSDC(std::string path = "");
  bool runLiberty(std::vector<std::string> paths);
  bool runSpef(std::string path = "");
  bool reportTiming();
  void buildNetGraph();
  double getPeriodNS(std::string clock_name);

 private:
  static StaIO* _instance;

  StaIO() {}
  ~StaIO() = default;

  void set_instance_flip_flop();
};

}  // namespace iplf

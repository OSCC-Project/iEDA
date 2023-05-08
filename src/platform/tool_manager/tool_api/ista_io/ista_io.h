/*
 * @Author: sjchanson 13560469332@163.com
 * @Date: 2022-12-03 10:55:45
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @LastEditTime: 2023-02-21 16:00:48
 * @FilePath: /irefactor/src/platform/tool_manager/tool_api/ista_io/ista_io.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置
 * 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#pragma once
/**
 * iEDA
 * Copyright (C) 2021  PCL
 *
 * This program is free software;
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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
  bool initSTA(std::string path = "");
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

 private:
  static StaIO* _instance;

  StaIO() {}
  ~StaIO() = default;

  bool setStaWorkDirectory(std::string path = "");
  bool readIdb(idb::IdbBuilder* idb_builder = nullptr);
  bool runSDC(std::string path = "");
  bool runLiberty(std::vector<std::string> paths);
  bool runSpef(std::string path = "");
  bool reportTiming();
};

}  // namespace iplf

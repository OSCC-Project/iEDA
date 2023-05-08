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
#include <map>
#include <string>
#include <vector>

#include "DrcViolationSpot.h"
#include "file_drc.h"

namespace iplf {

#define drcInst (DrcIO::getInstance())
class DrcIO
{
 public:
  static DrcIO* getInstance()
  {
    if (!_instance) {
      _instance = new DrcIO;
    }
    return _instance;
  }

  /// getter
  int32_t get_buffer_size();
  std::map<std::string, std::vector<idrc::DrcViolationSpot*>>& get_detail_drc() { return _detail_drc; }

  /// io
  bool runDRC(std::string config = "", std::string report_path = "");
  std::tuple<bool, std::vector<std::string>, std::vector<std::string>, int> checkConnnectivity();
  bool readDrcFromFile(std::string path = "");
  bool saveDrcToFile(std::string path);

  std::map<std::string, std::vector<idrc::DrcViolationSpot*>> getDetailCheckResult(std::string path = "");

 private:
  static DrcIO* _instance;

  std::map<std::string, std::vector<idrc::DrcViolationSpot*>> _detail_drc;

  DrcIO() {}
  ~DrcIO() = default;
};

}  // namespace iplf

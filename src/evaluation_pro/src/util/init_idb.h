/*
 * @FilePath: init_idb.h
 * @Author: Yihang Qiu (qiuyihang23@mails.ucas.ac.cn)
 * @Date: 2024-09-02 14:14:04
 * @Description:
 */
#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "congestion_db.h"

namespace ieval {

class InitIDB
{
 public:
  InitIDB();
  ~InitIDB();

  static InitIDB* getInst();
  static void destroyInst();

  // for wirelength evaluation
  void initPointSets();
  std::vector<std::vector<std::pair<int32_t, int32_t>>> getPointSets() { return _point_sets; }
  int32_t getDesignUnit();
  std::map<std::string, std::vector<std::pair<int32_t, int32_t>>> getNamePointSet() { return _name_point_set; }

  // for congestion evaluation
  void initCongestionDB();
  CongestionRegion getCongestionRegion() { return _region; }
  CongestionNets getCongestionNets() { return _nets; }

 private:
  static InitIDB* _init_idb;

  // for wirelength evaluation
  std::vector<std::vector<std::pair<int32_t, int32_t>>> _point_sets;
  std::map<std::string, std::vector<std::pair<int32_t, int32_t>>> _name_point_set;

  // for congestion evaluation
  CongestionRegion _region;
  CongestionNets _nets;
};
}  // namespace ieval

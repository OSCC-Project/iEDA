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
#include "density_db.h"

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
  int32_t getDesignUnit();
  int32_t getRowHeight();
  std::vector<std::vector<std::pair<int32_t, int32_t>>> getPointSets() { return _point_sets; }
  std::map<std::string, std::vector<std::pair<int32_t, int32_t>>> getNamePointSet() { return _name_point_set; }

  // for congestion evaluation
  void initCongestionDB();
  CongestionRegion getCongestionRegion() { return _congestion_region; }
  CongestionNets getCongestionNets() { return _congestion_nets; }

  // for density evaluation
  void initDensityDB();
  void initDensityDBRegion();
  void initDensityDBCells();
  void initDensityDBNets();
  DensityRegion getDensityRegion() { return _density_region; }
  DensityRegion getDensityRegionCore() { return _density_region_core; }
  DensityCells getDensityCells() { return _density_cells; }
  DensityPins getDensityPins() { return _density_pins; }
  DensityNets getDensityNets() { return _density_nets; }
  int32_t getDieHeight();
  int32_t getDieWidth();

 private:
  static InitIDB* _init_idb;

  // for wirelength evaluation
  std::vector<std::vector<std::pair<int32_t, int32_t>>> _point_sets;
  std::map<std::string, std::vector<std::pair<int32_t, int32_t>>> _name_point_set;

  // for congestion evaluation
  CongestionRegion _congestion_region;
  CongestionNets _congestion_nets;

  // for density evaluation
  DensityRegion _density_region;
  DensityRegion _density_region_core;
  DensityCells _density_cells;
  DensityPins _density_pins;
  DensityNets _density_nets;
  bool _density_db_initialized = false;
};
}  // namespace ieval

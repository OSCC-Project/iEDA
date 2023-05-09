/*
 * @Author: Shijian Chen  chenshj@pcl.ac.cn
 * @Date: 2023-03-01 17:06:44
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @LastEditTime: 2023-03-10 11:03:00
 * @FilePath: /irefactor/src/operation/iPL/source/module/detail_refactor/DetailPlacer.hh
 * @Description: Main for detail placement
 *
 *
 */
#ifndef IPL_DETAILPLACER_H
#define IPL_DETAILPLACER_H

#include "DPOperator.hh"
#include "GridManager.hh"
#include "PlacerDB.hh"
#include "TopologyManager.hh"
#include "database/DPDatabase.hh"

namespace ipl {

class DetailPlacer
{
 public:
  DetailPlacer() = delete;
  DetailPlacer(Config* pl_config, PlacerDB* placer_db);
  DetailPlacer(const DetailPlacer&) = delete;
  DetailPlacer(DetailPlacer&&) = delete;
  ~DetailPlacer();

  DetailPlacer& operator=(const DetailPlacer&) = delete;
  DetailPlacer& operator=(DetailPlacer&&) = delete;

  bool checkIsLegal();
  void runDetailPlace();
  int64_t calTotalHPWL();
  float calPeakBinDensity();

 private:
  DPConfig _config;
  DPDatabase _database;
  DPOperator _operator;

  void initDPConfig(Config* pl_config);
  void initDPDatabase(PlacerDB* placer_db);
  void initDPLayout();
  void wrapRowList();
  void wrapRegionList();
  void wrapCellList();
  void initDPDesign();
  void wrapInstanceList();
  DPInstance* wrapInstance(Instance* pl_inst);
  void wrapNetList();
  DPNet* wrapNet(Net* pl_net);
  DPPin* wrapPin(Pin* pl_pin);
  void updateInstanceList();
  void correctOutsidePinCoordi();
  void initIntervalList();

  void clearClusterInfo();
  void alignInstanceOrient();
};
}  // namespace ipl

#endif
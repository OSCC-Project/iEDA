/*
 * @Author: S.J Chen
 * @Date: 2022-03-06 14:45:25
 * @LastEditTime: 2023-03-03 10:07:55
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @Description:
 * @FilePath: /irefactor/src/operation/iPL/source/module/global_placer/analytical_placer/NesterovPlace.hh
 * Contact : https://github.com/sjchanson
 */

#ifndef IPL_OPERATOR_GP_NESTEROV_PLACE_H
#define IPL_OPERATOR_GP_NESTEROV_PLACE_H

#include <fstream>
#include <iostream>

#include "Config.hh"
#include "Log.hh"
#include "PlacerDB.hh"
#include "config/NesterovPlaceConfig.hh"
#include "database/NesterovDatabase.hh"

namespace ipl {

class NesterovPlace
{
 public:
  NesterovPlace() = delete;
  NesterovPlace(Config* config, PlacerDB* placer_db);
  NesterovPlace(const NesterovPlace&) = delete;
  NesterovPlace(NesterovPlace&&) = delete;
  ~NesterovPlace();

  NesterovPlace& operator=(const NesterovPlace&) = delete;
  NesterovPlace& operator=(NesterovPlace&&) = delete;

  void runNesterovPlace();
  void runNesterovRoutablityPlace();

  void printNesterovDatabase();

 private:
  NesterovPlaceConfig _nes_config;
  NesterovDatabase* _nes_database;

  void initNesConfig(Config* config);
  void initNesDatabase(PlacerDB* placer_db);
  void wrapNesInstanceList();
  void wrapNesInstance(Instance* inst, NesInstance* nesInst);
  void wrapNesNetList();
  void wrapNesNet(Net* net, NesNet* nesNet);
  void wrapNesPinList();
  void wrapNesPin(Pin* pin, NesPin* nesPin);
  void completeConnection();

  void initFillerNesInstance();
  void initNesInstanceDensitySize();

  void initNesterovPlace(std::vector<NesInstance*>& inst_list);
  void NesterovSolve(std::vector<NesInstance*>& inst_list);
  void NesterovRoutablitySolve(std::vector<NesInstance*>& inst_list);

  std::vector<NesInstance*> obtianPlacableNesInstanceList();

  void updateDensityCoordiLayoutInside(NesInstance* nInst, Rectangle<int32_t> core_shape);
  void updateDensityCenterCoordiLayoutInside(NesInstance* nInst, Point<int32_t>& center_coordi, Rectangle<int32_t> region_shape);

  void initGridManager();
  void initGridFixedArea();

  void initTopologyManager();
  void updateTopologyManager();

  void initBaseWirelengthCoef();
  void updateWirelengthCoef(float overflow);

  void updatePenaltyGradient(std::vector<NesInstance*>& nInst_list, std::vector<Point<float>>& sum_grads,
                             std::vector<Point<float>>& wirelength_grads, std::vector<Point<float>>& density_grads);

  Point<float> obtainWirelengthPrecondition(NesInstance* nInst);
  Point<float> obtainDensityPrecondition(NesInstance* nInst);

  Rectangle<int32_t> obtainFirstGridShape();
  int64_t obtainTotalArea(std::vector<NesInstance*>& inst_list);
  float obtainPhiCoef(float scaled_diff_hpwl);
  int64_t obtainTotalFillerArea(std::vector<NesInstance*>& inst_list);

  void writeBackPlacerDB();

  void updateNetWeight();

  // DEBUG.
  void printAcrossLongNet(std::ofstream& file_stream, int32_t max_width, int32_t max_height);
  void printIterationCoordi(std::ofstream& file_stream, int32_t cur_iter);
  void saveNesterovPlaceData(int32_t cur_iter);

  // Precondition Test
  std::vector<double> _global_diagonal_list;
  void initDiagonalIdentityMatrix(int32_t inst_size);
  void initDiagonalHkMatrix(std::vector<NesInstance*>& inst_list);
  void initDiagonalSkMatrix(std::vector<NesInstance*>& inst_list);
  void updatePenaltyGradientPre1(std::vector<NesInstance*>& nInst_list, std::vector<Point<float>>& sum_grads,
                                 std::vector<Point<float>>& wirelength_grads, std::vector<Point<float>>& density_grads);
  void updatePenaltyGradientPre2(std::vector<NesInstance*>& nInst_list, std::vector<Point<float>>& sum_grads,
                                 std::vector<Point<float>>& wirelength_grads, std::vector<Point<float>>& density_grads);
};
inline NesterovPlace::NesterovPlace(Config* config, PlacerDB* placer_db) : _nes_database(nullptr)
{
  initNesConfig(config);
  initNesDatabase(placer_db);
  initFillerNesInstance();
  initNesInstanceDensitySize();
}
inline NesterovPlace::~NesterovPlace()
{
  delete _nes_database;
}

}  // namespace ipl

#endif

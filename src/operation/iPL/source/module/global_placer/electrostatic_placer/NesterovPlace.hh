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
/*
 * @Author: S.J Chen
 * @Date: 2022-03-06 14:45:25
 * @LastEditTime: 2023-03-03 10:07:55
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @Description:
 * @FilePath: /irefactor/src/operation/iPL/source/module/global_placer/electrostatic_placer/NesterovPlace.hh
 * Contact : https://github.com/sjchanson
 */

#ifndef IPL_OPERATOR_GP_NESTEROV_PLACE_H
#define IPL_OPERATOR_GP_NESTEROV_PLACE_H

#include <float.h>

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
  NesterovPlace(Config* config, PlacerDB* placer_db, bool enableJsonOutput = false);
  NesterovPlace(const NesterovPlace&) = delete;
  NesterovPlace(NesterovPlace&&) = delete;
  ~NesterovPlace();

  NesterovPlace& operator=(const NesterovPlace&) = delete;
  NesterovPlace& operator=(NesterovPlace&&) = delete;

  void runNesterovPlace();
  void printNesterovDatabase();

  bool isJsonOutputEnabled() { return _enable_json_output; }

 private:
  NesterovPlaceConfig _nes_config;
  NesterovDatabase* _nes_database;

  // For convergence acceleration and non-convergence treatment
  int64_t _best_hpwl = INT64_MAX;
  float _best_overflow = FLT_MAX;
  std::vector<float> _overflow_record_list;
  std::vector<float> _hpwl_record_list;
  float _quad_penalty_coeff = 0.005;
  int64_t _total_inst_area = 0;
  bool _enable_json_output = false;

  void resetOverflowRecordList();
  void resetHPWLRecordList();
  void initQuadPenaltyCoeff();
  bool checkPlateau(int32_t window, float threshold);
  void entropyInjection(float shrink_factor, float noise_intensity);
  bool checkDivergence(int32_t window, float threshold, bool is_routability = false);
  bool checkLongTimeOverflowUnchanged(int32_t window, float threshold);

  void initNesConfig(Config* config);
  void calculateAdaptiveBinCnt();
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

  std::vector<NesInstance*> obtianPlacableNesInstanceList();

  void updateDensityCoordiLayoutInside(NesInstance* nInst, Rectangle<int32_t> core_shape);
  void updateDensityCenterCoordiLayoutInside(NesInstance* nInst, Point<int32_t>& center_coordi, Rectangle<int32_t> region_shape);

  void initGridManager();
  void initGridFixedArea();

  void initTopologyManager();
  void initNodes();
  void initNetWorks();
  void initGroups();
  void initArcs();
  void generatePortOutNetArc(Node* node);
  void generateNetArc(Node* node);
  void generateGroupArc(Node* node);
  void initHPWLEvaluator();
  void initWAWLGradientEvaluator();
  void initTimingAnnotation();
  void updateTopologyManager();

  void initBaseWirelengthCoef();
  void updateWirelengthCoef(float overflow);

  void updatePenaltyGradient(std::vector<NesInstance*>& nInst_list, std::vector<Point<float>>& sum_grads,
                             std::vector<Point<float>>& wirelength_grads, std::vector<Point<float>>& density_grads,
                             bool is_add_quad_penalty);

  Point<float> obtainWirelengthPrecondition(NesInstance* nInst);
  Point<float> obtainDensityPrecondition(NesInstance* nInst);

  Rectangle<int32_t> obtainFirstGridShape();
  int64_t obtainTotalArea(std::vector<NesInstance*>& inst_list);
  float obtainPhiCoef(float scaled_diff_hpwl, int32_t iteration_num);
  int64_t obtainTotalFillerArea(std::vector<NesInstance*>& inst_list);

  void writeBackPlacerDB();

  void updateMaxLengthNetWeight();
  void updateTimingNetWeight();

  // DEBUG.
  void printAcrossLongNet(std::ofstream& file_stream, int32_t max_width, int32_t max_height);
  void printIterationCoordi(std::ofstream& file_stream, int32_t cur_iter);
  void saveNesterovPlaceData(int32_t cur_iter);
  void plotInstImage(std::string file_name);
  void plotInstJson(std::string file_name, int32_t cur_iter, float overflow);
  void plotBinForceLine(std::string file_name);
  void printIterInfoToCsv(std::ofstream& file_stream, int32_t iter_num);
  void printDensityMapToCsv(std::string file_name);

  // Precondition Test
  std::vector<double> _global_diagonal_list;
  void initDiagonalIdentityMatrix(int32_t inst_size);
  void initDiagonalHkMatrix(std::vector<NesInstance*>& inst_list);
  void initDiagonalSkMatrix(std::vector<NesInstance*>& inst_list);
  void updatePenaltyGradientPre1(std::vector<NesInstance*>& nInst_list, std::vector<Point<float>>& sum_grads,
                                 std::vector<Point<float>>& wirelength_grads, std::vector<Point<float>>& density_grads);
  void updatePenaltyGradientPre2(std::vector<NesInstance*>& nInst_list, std::vector<Point<float>>& sum_grads,
                                 std::vector<Point<float>>& wirelength_grads, std::vector<Point<float>>& density_grads);
  
  void notifyPLBinSize();
  void notifyPLOverflowInfo(float final_overflow);
  void notifyPLPlaceDensity();
};
inline NesterovPlace::NesterovPlace(Config* config, PlacerDB* placer_db, bool enableJsonOutput)
    : _nes_database(nullptr), _enable_json_output(enableJsonOutput)
{
  initNesConfig(config);
  initNesDatabase(placer_db);
  initFillerNesInstance();
  initNesInstanceDensitySize();

  // init bin inst type
  _nes_database->_bin_grid->initNesInstanceTypeList(_nes_database->_nInstance_list);
}
inline NesterovPlace::~NesterovPlace()
{
  delete _nes_database;
}

}  // namespace ipl

#endif

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
 * @Date: 2022-10-27 16:17:57
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @LastEditTime: 2023-03-11 14:36:48
 * @FilePath: /irefactor/src/operation/iPL/api/PLAPI.cc
 * @Description:
 */
#include "PLAPI.hh"

#include <filesystem>

#include "BufferInserter.hh"
#include "CenterPlace.hh"
#include "DetailPlacer.hh"
#include "EvalAPI.hpp"
#include "IDBWrapper.hh"
#include "LayoutChecker.hh"
#include "Legalizer.hh"
#include "Log.hh"
#include "MacroPlacer.hh"
#include "NesterovPlace.hh"
#include "PlacerDB.hh"
#include "RandomPlace.hh"
#include "SteinerWirelength.hh"
#include "src/MapFiller.h"
#include "timing/TimingEval.hpp"
#include "tool_api/ista_io/ista_io.h"

namespace ipl {

// NOLINTBEGIN
eval::TimingPin* wrapTimingTruePin(Node* node);
eval::TimingPin* wrapTimingFakePin(int id, Point<int32_t> coordi);
eval::CongPin* wrapCongPin(ipl::Pin* ipl_pin);
// NOLINTEND

PLAPI& PLAPI::getInst()
{
  if (!_s_ipl_api_instance) {
    _s_ipl_api_instance = new PLAPI();
  }

  return *_s_ipl_api_instance;
}

void PLAPI::destoryInst()
{
  if (_s_ipl_api_instance->isAbucasLGStarted()) {
    LegalizerInst.destoryInst();
  }

  if (_s_ipl_api_instance->isPlacerDBStarted()) {
    PlacerDBInst.destoryInst();
  }

  if (_s_ipl_api_instance) {
    delete _s_ipl_api_instance;
  }
}

PLAPI::~PLAPI()
{
  // if (_timing_evaluator) {
  //   delete _timing_evaluator;
  // }
}

void PLAPI::initAPI(std::string pl_json_path, idb::IdbBuilder* idb_builder)
{
  char config[] = "info_ipl_glog";
  char* argv[] = {config};
  Log::init(argv);
  IDBWrapper* idb_wrapper = new IDBWrapper(idb_builder);
  PlacerDBInst.initPlacerDB(pl_json_path, idb_wrapper);

  // create log and report folder
  if (!std::filesystem::exists("./result/pl/log")) {
    if (std::filesystem::create_directory("./result/pl/log")) {
      LOG_INFO << "Create folder './result/pl/log' for iPL log";
    } else {
      LOG_ERROR << "Cannot create './result/pl/log' for iPL log";
    }
  }
  if (!std::filesystem::exists("./result/pl/report")) {
    if (std::filesystem::create_directory("./result/pl/report")) {
      LOG_INFO << "Create folder './result/pl/report' for iPL report";
    } else {
      LOG_ERROR << "Cannot create './result/pl/report' for iPL report";
    }
  }

  // prepare sta for timing aware mode placement
  if (PlacerDBInst.get_placer_config()->isTimingAwareMode()) {
    // sta has not been initialized
    if (!this->isSTAStarted()) {
      LOG_INFO << "Try to apply to start iSTA";
      // apply to start sta
      this->initSTA();
      this->updateSTATiming();
    }

    updateSequentialProperty();

    LOG_INFO << "Sucessfully update sequential property, the latest instance and net infomations are as follow: ";
    PlacerDBInst.printInstanceInfo();
    PlacerDBInst.printNetInfo();
  }
}

void PLAPI::runIncrementalFlow()
{
  runLG();
  reportPLInfo();
  writeBackSourceDataBase();
}

/*****************************Timing-driven Placement: Start*****************************/
void PLAPI::initTimingEval()
{
  eval::EvalAPI& eval_api = eval::EvalAPI::initInst();
  eval_api.initTimingEval(PlacerDBInst.get_layout()->get_database_unit());
}

double PLAPI::obtainPinEarlySlack(std::string pin_name)
{
  return eval::EvalAPI::getInst().getEarlySlack(pin_name);
}

double PLAPI::obtainPinLateSlack(std::string pin_name)
{
  return eval::EvalAPI::getInst().getLateSlack(pin_name);
}

double PLAPI::obtainPinEarlyArrivalTime(std::string pin_name)
{
  return eval::EvalAPI::getInst().getArrivalEarlyTime(pin_name);
}

double PLAPI::obtainPinLateArrivalTime(std::string pin_name)
{
  return eval::EvalAPI::getInst().getArrivalLateTime(pin_name);
}

double PLAPI::obtainPinEarlyRequiredTime(std::string pin_name)
{
  return eval::EvalAPI::getInst().getRequiredEarlyTime(pin_name);
}

double PLAPI::obtainPinLateRequiredTime(std::string pin_name)
{
  return eval::EvalAPI::getInst().getRequiredLateTime(pin_name);
}

double PLAPI::obtainWNS(const char* clock_name, ista::AnalysisMode mode)
{
  return eval::EvalAPI::getInst().reportWNS(clock_name, mode);
}

double PLAPI::obtainTNS(const char* clock_name, ista::AnalysisMode mode)
{
  return eval::EvalAPI::getInst().reportTNS(clock_name, mode);
}

void PLAPI::updateTiming()
{
  auto* topo_manager = PlacerDBInst.get_topo_manager();
  SteinerWirelength steiner_wl(topo_manager);
  steiner_wl.updateAllNetWorkPointPair();

  std::vector<eval::TimingNet*> timing_net_list;
  timing_net_list.reserve(topo_manager->get_network_list().size());
  for (auto* network : topo_manager->get_network_list()) {
    eval::TimingNet* timing_net = new eval::TimingNet();
    timing_net->set_name(network->get_name());
    std::map<Point<int32_t>, Node*, PointCMP> point_to_node;
    std::map<Point<int32_t>, eval::TimingPin*, PointCMP> point_to_timing_pin;
    for (auto* node : network->get_node_list()) {
      const auto& node_loc = node->get_location();
      auto iter = point_to_node.find(node_loc);
      if (iter != point_to_node.end()) {
        auto* timing_pin_1 = wrapTimingTruePin(node);
        auto* timing_pin_2 = wrapTimingTruePin(iter->second);
        timing_net->add_pin_pair(timing_pin_1, timing_pin_2);
      } else {
        point_to_node.emplace(node_loc, node);
      }
    }

    const auto& point_pair_list = steiner_wl.obtainPointPairList(network);
    int fake_pin_id = 0;
    for (auto point_pair : point_pair_list) {
      if (point_pair.first == point_pair.second) {
        continue;
      }
      eval::TimingPin* timing_pin_1 = nullptr;
      eval::TimingPin* timing_pin_2 = nullptr;
      auto iter_1 = point_to_node.find(point_pair.first);
      if (iter_1 != point_to_node.end()) {
        auto iter_1_1 = point_to_timing_pin.find(point_pair.first);
        if (iter_1_1 != point_to_timing_pin.end()) {
          timing_pin_1 = iter_1_1->second;
        } else {
          timing_pin_1 = wrapTimingTruePin(iter_1->second);
          point_to_timing_pin.emplace(point_pair.first, timing_pin_1);
        }
      } else {
        auto iter_1_2 = point_to_timing_pin.find(point_pair.first);
        if (iter_1_2 != point_to_timing_pin.end()) {
          timing_pin_1 = iter_1_2->second;
        } else {
          timing_pin_1 = wrapTimingFakePin(fake_pin_id++, point_pair.first);
          point_to_timing_pin.emplace(point_pair.first, timing_pin_1);
        }
      }
      auto iter_2 = point_to_node.find(point_pair.second);
      if (iter_2 != point_to_node.end()) {
        auto iter_2_1 = point_to_timing_pin.find(point_pair.second);
        if (iter_2_1 != point_to_timing_pin.end()) {
          timing_pin_2 = iter_2_1->second;
        } else {
          timing_pin_2 = wrapTimingTruePin(iter_2->second);
          point_to_timing_pin.emplace(point_pair.second, timing_pin_2);
        }
      } else {
        auto iter_2_2 = point_to_timing_pin.find(point_pair.second);
        if (iter_2_2 != point_to_timing_pin.end()) {
          timing_pin_2 = iter_2_2->second;
        } else {
          timing_pin_2 = wrapTimingFakePin(fake_pin_id++, point_pair.second);
          point_to_timing_pin.emplace(point_pair.second, timing_pin_2);
        }
      }
      timing_net->add_pin_pair(timing_pin_1, timing_pin_2);
    }
    timing_net_list.push_back(timing_net);
  }
  EvalInst.updateTiming(timing_net_list);
}

void PLAPI::updateTimingInstMovement(std::map<std::string, std::vector<std::pair<Point<int32_t>, Point<int32_t>>>> influenced_net_map,
                                     std::vector<std::string> moved_inst_list)
{
}

void PLAPI::destroyTimingEval()
{
  eval::EvalAPI::destroyInst();
}
/*****************************Timing-driven Placement: END*****************************/

void PLAPI::runFlow()
{
  // runMP();
  runGP();
  printHPWLInfo();

  if (PlacerDBInst.get_placer_config()->get_buffer_config().isMaxLengthOpt()) {
    std::cout << std::endl;
    runBufferInsertion();
    printHPWLInfo();
  }

  std::cout << std::endl;
  runLG();
  printHPWLInfo();

  std::cout << std::endl;
  runDP();
  printHPWLInfo();

  std::cout << std::endl;

  // // update LG Database
  // LOG_INFO << "Repeated execution of legalization to support subsequent incremental legalization";
  // LegalizerInst.updateInstanceList();
  // LegalizerInst.runLegalize();
  // printHPWLInfo();
  // std::cout << std::endl;

  reportPLInfo();
  std::cout << std::endl;
  LOG_INFO << "Log has been writed to dir: ./result/pl/log/";

  writeBackSourceDataBase();
}

void PLAPI::insertLayoutFiller()
{
  MapFiller(&PlacerDBInst, PlacerDBInst.get_placer_config()).mapFillerCell();
  PlacerDBInst.updateGridManager();
  reportPLInfo();
  reportLayoutWhiteInfo();
  writeBackSourceDataBase();
}

void PLAPI::runMP()
{
  imp::MPDB* mpdb = new imp::MPDB(&PlacerDBInst);
  imp::MacroPlacer(mpdb, PlacerDBInst.get_placer_config()).runMacroPlacer();
  delete mpdb;
}

void PLAPI::runGP()
{
  // CenterPlace(&PlacerDBInst).runCenterPlace();
  RandomPlace(&PlacerDBInst).runRandomPlace();
  NesterovPlace nesterov_place(PlacerDBInst.get_placer_config(), &PlacerDBInst);
  nesterov_place.printNesterovDatabase();
  nesterov_place.runNesterovPlace();
}

bool PLAPI::runLG()
{
  LegalizerInst.initLegalizer(PlacerDBInst.get_placer_config(), &PlacerDBInst);
  bool flag = LegalizerInst.runLegalize();
  LOG_ERROR_IF(!flag) << "Legalization is not completed!";
  return flag;
}

bool PLAPI::runIncrLG()
{
  PlacerDBInst.updateFromSourceDataBase();
  LegalizerInst.updateInstanceList();
  bool flag = LegalizerInst.runIncrLegalize();
  return flag;
}

bool PLAPI::runIncrLG(std::vector<std::string> inst_name_list)
{
  auto* design = PlacerDBInst.get_design();
  std::vector<Instance*> inst_list;
  for (std::string inst_name : inst_name_list) {
    auto* inst = design->find_instance(inst_name);
    inst_list.push_back(inst);
  }

  LegalizerInst.updateInstanceList(inst_list);
  bool flag = LegalizerInst.runIncrLegalize();

  return flag;
}

void PLAPI::runDP()
{
  bool legal_flag = checkLegality();
  if (!legal_flag) {
    LOG_WARNING << "Design Instances before detail placement are not legal";
    return;
  }

  DetailPlacer detail_place(PlacerDBInst.get_placer_config(), &PlacerDBInst);
  detail_place.runDetailPlace();

  if (!checkLegality()) {
    LOG_WARNING << "DP result is not legal";
  }
}

void PLAPI::initSTA()
{
  staInst->initSTA();
  staInst->buildGraph();
}
void PLAPI::updateSTATiming()
{
  staInst->updateTiming();
}

bool PLAPI::isClockNet(std::string net_name)
{
  bool flag = staInst->isClockNet(net_name);
  return flag;
}

bool PLAPI::isSequentialCell(std::string inst_name)
{
  return staInst->isSequentialCell(inst_name);

  bool is_sequential = false;
  auto* inst = PlacerDBInst.get_design()->find_instance(inst_name);
  for (auto* pin : inst->get_pins()) {
    auto* net = pin->get_net();
    if (isClockNet(net->get_name())) {
      is_sequential = true;
      break;
    }
  }
  return is_sequential;
}

bool PLAPI::isBufferCell(std::string cell_name)
{
  std::string cell_type = staInst->getCellType(cell_name.c_str());
  bool flag = (cell_type == "Buffer");
  return flag;
}

void PLAPI::updateSequentialProperty()
{
  for (auto* inst : PlacerDBInst.get_design()->get_instance_list()) {
    if (inst->isOutsideInstance()) {
      continue;
    }

    if (!inst->get_cell_master()) {
      continue;
    }

    auto* cell_master = inst->get_cell_master();

    if (cell_master->isMacro()) {
      continue;
    }

    if (cell_master->isPhysicalFiller()) {
      continue;
    }

    if (isSequentialCell(inst->get_name())) {
      inst->get_cell_master()->set_type(CELL_TYPE::kFlipflop);
    }

    // buffer identification
    if (isBufferCell(cell_master->get_name())) {
      inst->get_cell_master()->set_type(CELL_TYPE::kLogicBuffer);
    }
  }

  for (auto* net : PlacerDBInst.get_design()->get_net_list()) {
    if (this->isClockNet(net->get_name())) {
      net->set_net_type(NET_TYPE::kClock);
      net->set_netweight(0.0f);
      net->set_net_state(NET_STATE::kDontCare);
    }
  }

  // clock buffer identification
  for (auto* inst : PlacerDBInst.get_design()->get_instance_list()) {
    if (inst->get_cell_master() && inst->get_cell_master()->isLogicBuffer()) {
      bool is_sequential = false;
      for (auto* pin : inst->get_pins()) {
        if (pin->get_net()->isClockNet()) {
          is_sequential = true;
          break;
        }
      }

      if (is_sequential) {
        inst->get_cell_master()->set_type(CELL_TYPE::kClockBuffer);
      }
    }
  }
}

std::vector<std::string> PLAPI::obtainClockNameList()
{
  return staInst->getClockNameList();
}

void PLAPI::runBufferInsertion()
{
  BufferInserter buffer_inserter(PlacerDBInst.get_placer_config(), &PlacerDBInst);
  buffer_inserter.runBufferInsertionForMaxWireLength();
}

void PLAPI::updatePlacerDB()
{
  PlacerDBInst.updateFromSourceDataBase();
}

void PLAPI::updatePlacerDB(std::vector<std::string> inst_list)
{
  PlacerDBInst.updateFromSourceDataBase(inst_list);
}

bool PLAPI::insertSignalBuffer(std::pair<std::string, std::string> source_sink_net, std::vector<std::string> sink_pin_list,
                               std::pair<std::string, std::string> master_inst_buffer, std::pair<int, int> buffer_center_loc)
{
  bool flag = staInst->insertBuffer(source_sink_net, sink_pin_list, master_inst_buffer, buffer_center_loc, idb::IdbConnectType::kSignal);
  return flag;
}

void PLAPI::writeBackSourceDataBase()
{
  PlacerDBInst.writeBackSourceDataBase();
}

std::vector<Rectangle<int32_t>> PLAPI::obtainAvailableWhiteSpaceList(std::pair<int32_t, int32_t> row_range,
                                                                     std::pair<int32_t, int32_t> site_range)
{
  assert(row_range.first < row_range.second && site_range.first < site_range.second);

  auto* grid_manager = PlacerDBInst.get_grid_manager();
  int32_t row_num = grid_manager->get_grid_cnt_y();
  int32_t site_num = grid_manager->get_grid_cnt_x();

  row_range.first < 0 ? row_range.first = 0 : row_range.first;
  row_range.second >= row_num ? row_range.second = (row_num - 1) : row_range.second;
  site_range.first < 0 ? site_range.first = 0 : site_range.first;
  site_range.second >= site_num ? site_range.second = (site_num - 1) : site_range.second;

  return grid_manager->obtainAvailableRectList(row_range.first, row_range.second, site_range.first, site_range.second, 1.0);
}

bool PLAPI::checkLegality()
{
  bool legal_flag = true;

  LayoutChecker checker(&PlacerDBInst);
  if (!checker.isAllPlacedInstInsideCore()) {
    legal_flag = false;
  }
  if (!checker.isAllPlacedInstAlignRowSite()) {
    legal_flag = false;
  }
  if (!checker.isAllPlacedInstAlignPower()) {
    legal_flag = false;
  }
  if (!checker.isNoOverlapAmongInsts()) {
    legal_flag = false;
  }

  return legal_flag;
}

bool PLAPI::isSTAStarted()
{
  return staInst->isInitSTA();
}

bool PLAPI::isPlacerDBStarted()
{
  return PlacerDBInst.isInitialized();
}

bool PLAPI::isAbucasLGStarted()
{
  // return AbacusLegalizerInst.isInitialized();
  return LegalizerInst.isInitialized();
}

/*****************************Congestion-driven Placement: START*****************************/
void PLAPI::runRoutabilityGP()
{
  CenterPlace(&PlacerDBInst).runCenterPlace();
  NesterovPlace nesterov_place(PlacerDBInst.get_placer_config(), &PlacerDBInst);
  nesterov_place.printNesterovDatabase();
  nesterov_place.runNesterovRoutablityPlace();
}

/**
 * @brief run GR based on dmInst data, evaluate 3D congestion, and return <ACE,TOF,MOF> vector
 * @return std::vector<float>
 */
std::vector<float> PLAPI::evalGRCong()
{
  eval::EvalAPI& eval_api = eval::EvalAPI::initInst();

  std::vector<float> gr_congestion;
  gr_congestion = eval_api.evalGRCong();

  return gr_congestion;
}

/**
 * @brief compute each gcellgrid routing demand/resource, and return a 2D route util map
 * @return std::vector<float>
 */
std::vector<float> PLAPI::getUseCapRatioList()
{
  eval::EvalAPI& eval_api = eval::EvalAPI::getInst();
  return eval_api.getUseCapRatioList();
}

/**
 * @brief draw congesiton map based on GR result
 * @param  plot_path
 * @param  output_file_name
 */
void PLAPI::plotCongMap(const std::string& plot_path, const std::string& output_file_name)
{
  eval::EvalAPI& eval_api = eval::EvalAPI::getInst();
  // layer by layer
  eval_api.plotGRCong(plot_path, output_file_name);
  // statistical TotalOverflow/MaximumOverflow
  eval_api.plotOverflow(plot_path, output_file_name);
}

void PLAPI::destroyCongEval()
{
  eval::EvalAPI::destroyInst();
}

std::vector<float> PLAPI::obtainPinDens()
{
  return eval::EvalAPI::getInst().evalPinDens();
}

std::vector<float> PLAPI::obtainNetCong(std::string rudy_type)
{
  return eval::EvalAPI::getInst().evalNetCong(rudy_type);
}
/*****************************Congestion-driven Placement: END*****************************/

eval::TimingPin* wrapTimingTruePin(Node* node)
{
  eval::TimingPin* timing_pin = new eval::TimingPin();
  timing_pin->set_name(node->get_name());
  timing_pin->set_coord(eval::Point<int64_t>(node->get_location().get_x(), node->get_location().get_y()));
  timing_pin->set_is_real_pin(true);

  return timing_pin;
}

eval::TimingPin* wrapTimingFakePin(int id, Point<int32_t> coordi)
{
  eval::TimingPin* timing_pin = new eval::TimingPin();
  timing_pin->set_name("fake_" + std::to_string(id));
  timing_pin->set_id(id);
  timing_pin->set_coord(eval::Point<int64_t>(coordi.get_x(), coordi.get_y()));
  timing_pin->set_is_real_pin(false);

  return timing_pin;
}

eval::CongPin* wrapCongPin(ipl::Pin* ipl_pin)
{
  eval::CongPin* cong_pin = new eval::CongPin();
  cong_pin->set_name(ipl_pin->get_name());
  int64_t x = ipl_pin->get_center_coordi().get_x();
  int64_t y = ipl_pin->get_center_coordi().get_y();
  cong_pin->set_x(x);
  cong_pin->set_y(y);
  cong_pin->set_coord(eval::Point<int64_t>(x, y));
  return cong_pin;
}

// private
PLAPI* PLAPI::_s_ipl_api_instance = nullptr;

}  // namespace ipl

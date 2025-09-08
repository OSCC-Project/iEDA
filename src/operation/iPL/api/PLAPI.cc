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
#include "IDBWrapper.hh"
#include "LayoutChecker.hh"
#include "Legalizer.hh"
#include "Log.hh"
// #include "MacroPlacer.hh"
#include "NesterovPlace.hh"
#include "PlacerDB.hh"
#include "PostGP.hh"
#include "RandomPlace.hh"
#include "SteinerWirelength.hh"
#include "congestion_db.h"
#include "feature_ipl.h"
#include "netlist/Net.hh"
#include "src/MapFiller.h"
#include "timing_db.hh"
#include "wirelength_db.h"
namespace ipl {

// NOLINTBEGIN
ieval::TimingPin* wrapTimingTruePin(Node* node);
ieval::TimingPin* wrapTimingFakePin(int id, Point<int32_t> coordi);
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
    _s_ipl_api_instance = nullptr;
  }
}

PLAPI::~PLAPI()
{
  delete _external_api;
  delete _reporter;
}

void PLAPI::initAPI(std::string pl_json_path, idb::IdbBuilder* idb_builder)
{
  _external_api = new ExternalAPI();
  _reporter = new PLReporter(_external_api);

  // create external_api and reporter ptr
  createPLDirectory();

  char config[] = "info_ipl_glog";
  char* argv[] = {config};

  std::string log_home_path = this->obtainTargetDir() + "/pl/log/";
  // std::string design_name = idb_builder->get_def_service()->get_design()->get_design_name();
  // std::string home_path = "./evaluation_task/benchmark/" + design_name + "/pl_reports/";

  Log::makeSureDirectoryExist(log_home_path);
  Log::init(argv, log_home_path);
  IDBWrapper* idb_wrapper = new IDBWrapper(idb_builder);
  PlacerDBInst.initPlacerDB(pl_json_path, idb_wrapper);

  // prepare sta for timing aware mode placement
  if (PlacerDBInst.get_placer_config()->isTimingEffort()) {
    // sta has not been initialized
    if (!this->isSTAStarted()) {
      LOG_INFO << "Try to apply to start iSTA";
      // apply to start sta
      std::string sta_home_path = this->obtainTargetDir() + "/sta";
      this->initSTA(sta_home_path, false);

      // tmp for evalution.
      // std::string design_name = PlacerDBInst.get_design()->get_design_name();
      // std::string sta_path = "./evaluation_task/benchmark/" + design_name + "/sta_reports/sta";
      // this->modifySTAOutputDir(sta_path);

      this->updateSTATiming();
      PlacerDBInst.get_topo_manager()->updateALLNodeTopoId();
    }

    updateSequentialProperty();

    LOG_INFO << "Sucessfully update sequential property, the latest instance and net infomations are as follow: ";
    PlacerDBInst.printInstanceInfo();
    PlacerDBInst.printNetInfo();
  }
}

void PLAPI::createPLDirectory()
{
  std::string pl_dir = this->obtainTargetDir();
  if (pl_dir == "") {
    pl_dir = ".";
  }
  // create log and report folder
  if (!std::filesystem::exists(pl_dir + "/pl/log")) {
    if (std::filesystem::create_directories(pl_dir + "/pl/log")) {
      LOG_INFO << "Create folder " + pl_dir + "/pl/log for iPL log";
    } else {
      LOG_ERROR << "Cannot create " + pl_dir + "/pl/log for iPL log";
    }
  }
  if (!std::filesystem::exists(pl_dir + "/pl/report")) {
    if (std::filesystem::create_directories(pl_dir + "/pl/report")) {
      LOG_INFO << "Create folder " + pl_dir + "/pl/report for iPL report";
    } else {
      LOG_ERROR << "Cannot create " + pl_dir + "/pl/report for iPL report";
    }
  }
  if (!std::filesystem::exists(pl_dir + "/pl/plot")) {
    if (std::filesystem::create_directories(pl_dir + "/pl/plot")) {
      LOG_INFO << "Create folder " + pl_dir + "/pl/plot for iPL plot";
    } else {
      LOG_ERROR << "Cannot create " + pl_dir + "/pl/plot for iPL plot";
    }
  }
  if (!std::filesystem::exists(pl_dir + "/pl/gui")) {
    if (std::filesystem::create_directories(pl_dir + "/pl/gui")) {
      LOG_INFO << "Create folder " + pl_dir + "/pl/gui for iPL gui";
    } else {
      LOG_ERROR << "Cannot create " + pl_dir + "/pl/gui for iPL gui";
    }
  }
  if (!std::filesystem::exists(pl_dir + "/pl/density")) {
    if (std::filesystem::create_directories(pl_dir + "/pl/density")) {
      LOG_INFO << "Create folder " + pl_dir + "/pl/density for iPL density map";
    } else {
      LOG_ERROR << "Cannot create " + pl_dir + "/pl/density for iPL density map";
    }
  }
}

void PLAPI::runIncrementalFlow()
{
  runLG();
  notifyPLWLInfo(1);
  reportPLInfo();
  writeBackSourceDataBase();
}

/*****************************Timing-driven Placement: Start*****************************/
double PLAPI::obtainPinEarlySlack(std::string pin_name)
{
  return _external_api->obtainPinEarlySlack(pin_name);
}

double PLAPI::obtainPinLateSlack(std::string pin_name)
{
  return _external_api->obtainPinLateSlack(pin_name);
}

double PLAPI::obtainPinEarlyArrivalTime(std::string pin_name)
{
  return _external_api->obtainPinEarlyArrivalTime(pin_name);
}

double PLAPI::obtainPinLateArrivalTime(std::string pin_name)
{
  return _external_api->obtainPinLateArrivalTime(pin_name);
}

double PLAPI::obtainPinEarlyRequiredTime(std::string pin_name)
{
  return _external_api->obtainPinEarlyRequiredTime(pin_name);
}

double PLAPI::obtainPinLateRequiredTime(std::string pin_name)
{
  return _external_api->obtainPinLateRequiredTime(pin_name);
}

double PLAPI::obtainWNS(const char* clock_name, ista::AnalysisMode mode)
{
  return _external_api->obtainWNS(clock_name, mode);
}

double PLAPI::obtainTNS(const char* clock_name, ista::AnalysisMode mode)
{
  return _external_api->obtainTNS(clock_name, mode);
}

double PLAPI::obtainEarlyWNS(const char* clock_name)
{
  return _external_api->obtainWNS(clock_name, ista::AnalysisMode::kMin);
}

double PLAPI::obtainEarlyTNS(const char* clock_name)
{
  return _external_api->obtainTNS(clock_name, ista::AnalysisMode::kMin);
}

double PLAPI::obtainLateWNS(const char* clock_name)
{
  return _external_api->obtainWNS(clock_name, ista::AnalysisMode::kMax);
}

double PLAPI::obtainLateTNS(const char* clock_name)
{
  return _external_api->obtainTNS(clock_name, ista::AnalysisMode::kMax);
}

void PLAPI::updateTiming(TopologyManager* topo_manager)
{
  SteinerWirelength steiner_wl(topo_manager);
  steiner_wl.updateAllNetWorkPointPair();

  std::vector<ieval::TimingNet*> timing_net_list;
  timing_net_list.reserve(topo_manager->get_network_list().size());
  for (auto* network : topo_manager->get_network_list()) {
    const auto& point_pair_list = steiner_wl.obtainPointPairList(network);
    ieval::TimingNet* timing_net = generateTimingNet(network, point_pair_list);
    timing_net_list.push_back(timing_net);
  }
  _external_api->updateEvalTiming(timing_net_list, PlacerDBInst.get_layout()->get_database_unit());
}

void PLAPI::updatePartOfTiming(TopologyManager* topo_manager,
                               std::map<int32_t, std::vector<std::pair<Point<int32_t>, Point<int32_t>>>>& net_id_to_points_map)
{
  std::vector<ieval::TimingNet*> timing_net_list;
  timing_net_list.reserve(net_id_to_points_map.size());

  for (auto net_pair : net_id_to_points_map) {
    NetWork* network = topo_manager->findNetworkById(net_pair.first);
    ieval::TimingNet* timing_net = generateTimingNet(network, net_pair.second);
    timing_net_list.push_back(timing_net);
  }

  _external_api->updateEvalTiming(timing_net_list, PlacerDBInst.get_layout()->get_database_unit());
}

void PLAPI::updateTimingInstMovement(TopologyManager* topo_manager,
                                     std::map<int32_t, std::vector<std::pair<Point<int32_t>, Point<int32_t>>>> net_id_to_points_map,
                                     std::vector<std::string> moved_inst_list)
{
  std::vector<ieval::TimingNet*> timing_net_list;
  timing_net_list.reserve(net_id_to_points_map.size());

  for (auto net_pair : net_id_to_points_map) {
    NetWork* network = topo_manager->findNetworkById(net_pair.first);
    ieval::TimingNet* timing_net = generateTimingNet(network, net_pair.second);
    timing_net_list.push_back(timing_net);
  }

  _external_api->updateEvalTiming(timing_net_list, moved_inst_list, 3, PlacerDBInst.get_layout()->get_database_unit());
}

float PLAPI::obtainPinCap(std::string inst_pin_name)
{
  return _external_api->obtainPinCap(inst_pin_name);
}

float PLAPI::obtainAvgWireResUnitLengthUm()
{
  return _external_api->obtainAvgWireResUnitLengthUm();
}

float PLAPI::obtainAvgWireCapUnitLengthUm()
{
  return _external_api->obtainAvgWireCapUnitLengthUm();
}

float PLAPI::obtainInstOutPinRes(std::string inst_name, std::string port_name)
{
  auto* inst = PlacerDBInst.get_design()->find_instance(inst_name);
  auto* cell = inst->get_cell_master();
  std::string cell_name = cell->get_name();
  return _external_api->obtainInstOutPinRes(cell_name, port_name);
}

ieval::TimingNet* PLAPI::generateTimingNet(NetWork* network,
                                           const std::vector<std::pair<ipl::Point<int32_t>, ipl::Point<int32_t>>>& point_pair_list)
{
  ieval::TimingNet* timing_net = new ieval::TimingNet();
  timing_net->net_name = network->get_name();
  std::map<Point<int32_t>, Node*, PointCMP> point_to_node;
  std::map<Point<int32_t>, ieval::TimingPin*, PointCMP> point_to_timing_pin;
  for (auto* node : network->get_node_list()) {
    const auto& node_loc = node->get_location();
    auto iter = point_to_node.find(node_loc);
    if (iter != point_to_node.end()) {
      auto* timing_pin_1 = wrapTimingTruePin(node);
      auto* timing_pin_2 = wrapTimingTruePin(iter->second);
      timing_net->pin_pair_list.push_back(std::make_pair(timing_pin_1, timing_pin_2));
    } else {
      point_to_node.emplace(node_loc, node);
    }
  }

  int fake_pin_id = 0;
  for (auto point_pair : point_pair_list) {
    if (point_pair.first == point_pair.second) {
      continue;
    }
    ieval::TimingPin* timing_pin_1 = nullptr;
    ieval::TimingPin* timing_pin_2 = nullptr;
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
    timing_net->pin_pair_list.push_back(std::make_pair(timing_pin_1, timing_pin_2));
  }
  return timing_net;
}

void PLAPI::destroyTimingEval()
{
  _external_api->destroyTimingEval();
}
/*****************************Timing-driven Placement: END*****************************/

void PLAPI::runFlow()
{
  // runMP();
  runGP();
  // printHPWLInfo();
  // printTimingInfo();
  notifyPLWLInfo(0);
  // if (isSTAStarted()) {
  //   notifyPLCongestionInfo(0);
  //   notifyPLTimingInfo(0);
  // }

  if (PlacerDBInst.get_placer_config()->get_buffer_config().isMaxLengthOpt()) {
    std::cout << std::endl;
    runBufferInsertion();
    printHPWLInfo();
  }

  if (PlacerDBInst.get_placer_config()->get_dp_config().isEnableNetworkflow()) {
    std::cout << std::endl;
    runNetworkFlowSpread();
  }

  std::cout << std::endl;
  runLG();
  // printHPWLInfo();
  // printTimingInfo();
  notifyPLWLInfo(1);
  // if (isSTAStarted()) {
  //   notifyPLCongestionInfo(1);
  //   notifyPLTimingInfo(1);
  // }

  std::cout << std::endl;
  if (isSTAStarted()) {
    runPostGP();
  } else {
    // runDP(); // remove DP
  }
  // printHPWLInfo();
  // printTimingInfo();

  notifyPLWLInfo(2);
  // if (isSTAStarted()) {
  //   notifyPLCongestionInfo(2);
  //   notifyPLTimingInfo(2);
  // }

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

  // if (isSTAStarted()) {
  //   // notifySTAUpdateTimingRuntime();
  //   _reporter->reportTDPEvaluation();
  // }

  if (isSTAStarted()) {
    _external_api->destroyTimingEval();
  }

  writeBackSourceDataBase();
}

void PLAPI::runAiFlow(const std::string& onnx_path, const std::string& normalization_path)
{
  runGP();
  notifyPLWLInfo(0);

  if (PlacerDBInst.get_placer_config()->get_buffer_config().isMaxLengthOpt()) {
    std::cout << std::endl;
    runBufferInsertion();
    printHPWLInfo();
  }

  if (PlacerDBInst.get_placer_config()->get_dp_config().isEnableNetworkflow()) {
    std::cout << std::endl;
    runNetworkFlowSpread();
  }

  std::cout << std::endl;
  runLG();
  notifyPLWLInfo(1);

  std::cout << std::endl;
  if (isSTAStarted()) {
    runPostGP();
  } else {
#ifdef ENABLE_AI
    runDPwithAiWireLengthPredictor(onnx_path, normalization_path);
#else
    runDP();
#endif
  }
  notifyPLWLInfo(2);

  std::cout << std::endl;

  reportPLInfo();
  std::cout << std::endl;
  LOG_INFO << "Log has been writed to dir: ./result/pl/log/";

  if (isSTAStarted()) {
    _external_api->destroyTimingEval();
  }

  writeBackSourceDataBase();
}

void PLAPI::insertLayoutFiller()
{
  notifyPLOriginInfo();
  MapFiller(&PlacerDBInst, PlacerDBInst.get_placer_config()).mapFillerCell();
  PlacerDBInst.updateGridManager();
  _reporter->reportEDAFillerEvaluation();
  reportPLInfo();
  reportLayoutWhiteInfo();
  writeBackSourceDataBase();
}

// void PLAPI::runMP()
// {
//   imp::MPDB* mpdb = new imp::MPDB(&PlacerDBInst);
//   imp::MacroPlacer(mpdb, PlacerDBInst.get_placer_config()).runMacroPlacer();
//   delete mpdb;
// }

void PLAPI::runGP()
{
  // CenterPlace(&PlacerDBInst).runCenterPlace();
  RandomPlace(&PlacerDBInst).runRandomPlace();
  NesterovPlace nesterov_place(PlacerDBInst.get_placer_config(), &PlacerDBInst, isJsonOutputEnabled());
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

void PLAPI::runPostGP()
{
  //
  PostGP post_gp(PlacerDBInst.get_placer_config(), &PlacerDBInst);
  post_gp.runIncrTimingPlace();
}

bool PLAPI::runIncrLG()
{
  PlacerDBInst.updateFromSourceDataBase();
  LegalizerInst.updateInstanceList();
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

#ifdef ENABLE_AI
void PLAPI::runDPwithAiWireLengthPredictor(const std::string& onnx_path, const std::string& normalization_path)
{
  bool legal_flag = checkLegality();
  if (!legal_flag) {
    LOG_WARNING << "Design Instances before detail placement are not legal";
    return;
  }

  DetailPlacer detail_place(PlacerDBInst.get_placer_config(), &PlacerDBInst);

  if (!detail_place.init_ai_wirelength_model(onnx_path, normalization_path)) {
    LOG_ERROR << "Failed to load AI wirelength model: " << onnx_path;
    LOG_ERROR << "Falling back to traditional HPWL";
    return;
  }

  detail_place.runDetailPlace();

  if (!checkLegality()) {
    LOG_WARNING << "DP result is not legal";
  }
}
#endif

// run networkflow to spread cell
// Input: after global placement. Output: low density distribution result with overlap.
// Legalization is further needed.
void PLAPI::runNetworkFlowSpread()
{
  DetailPlacer detail_place(PlacerDBInst.get_placer_config(), &PlacerDBInst);
  detail_place.runDetailPlaceNFS();
}

void PLAPI::notifyPLWLInfo(int stage)
{
  // 1. origin method
  HPWirelength hpwl(PlacerDBInst.get_topo_manager());
  SteinerWirelength stwl(PlacerDBInst.get_topo_manager());
  stwl.updateAllNetWorkPointPair();
  PlacerDBInst.PL_HPWL[stage] = hpwl.obtainTotalWirelength();
  PlacerDBInst.PL_STWL[stage] = stwl.obtainTotalWirelength();

  // 2. most work in evalpro
  // this->writeBackSourceDataBase();
  // ieval::TotalWLSummary wl_summary = _external_api->evalproIDBWL();
  // PlacerDBInst.PL_HPWL[stage] = wl_summary.HPWL;
  // PlacerDBInst.PL_STWL[stage] = wl_summary.FLUTE;
  // PlacerDBInst.PL_GRWL[stage] = wl_summary.GRWL;
}

void PLAPI::notifyPLCongestionInfo(int stage)
{
  this->writeBackSourceDataBase();
  ieval::OverflowSummary overflow_summary = _external_api->evalproCongestion();
  PlacerDBInst.egr_tof[stage] = overflow_summary.total_overflow_union;
  PlacerDBInst.egr_mof[stage] = overflow_summary.max_overflow_union;
  PlacerDBInst.egr_ace[stage] = overflow_summary.weighted_average_overflow_union;

  float peak_average_pin_dens = _external_api->obtainPeakAvgPinDens();
  PlacerDBInst.pin_density[stage] = peak_average_pin_dens;
}

void PLAPI::notifyPLTimingInfo(int stage)
{
  this->updateTiming(PlacerDBInst.get_topo_manager());
  std::string clock_name = obtainClockNameList().at(0);
  float tns = obtainTNS(clock_name.c_str(), ista::AnalysisMode::kMax);
  float wns = obtainWNS(clock_name.c_str(), ista::AnalysisMode::kMax);

  PlacerDBInst.tns[stage] = tns;
  PlacerDBInst.wns[stage] = wns;

  float suggest_freq = 1000.0 / (_external_api->obtainTargetClockPeriodNS(clock_name) - wns);
  PlacerDBInst.suggest_freq[stage] = suggest_freq;
}

void PLAPI::notifySTAUpdateTimingRuntime()
{
  ieda::Stats sta_status;
  _external_api->updateSTATiming();
  double time_delta = sta_status.elapsedRunTime();
  PlacerDBInst.sta_update_time = time_delta;
}

void PLAPI::notifyPLOriginInfo()
{
  PlacerDBInst.init_inst_cnt = PlacerDBInst.get_design()->get_instance_list().size();
}

void PLAPI::modifySTAOutputDir(std::string path)
{
  _external_api->modifySTAOutputDir(path);
}

void PLAPI::initSTA(std::string path, bool init_log)
{
  _external_api->initSTA(path, init_log);
}

void PLAPI::updateSTATiming()
{
  _external_api->updateSTATiming();
}

bool PLAPI::isClockNet(std::string net_name)
{
  return _external_api->isClockNet(net_name);
}

bool PLAPI::isSequentialCell(std::string inst_name)
{
  return _external_api->isSequentialCell(inst_name);
}

bool PLAPI::isBufferCell(std::string cell_name)
{
  return _external_api->isBufferCell(cell_name);
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
  return _external_api->obtainClockNameList();
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
  return _external_api->insertSignalBuffer(source_sink_net, sink_pin_list, master_inst_buffer, buffer_center_loc);
}

void PLAPI::writeBackSourceDataBase()
{
  PlacerDBInst.writeBackSourceDataBase();
}

std::string PLAPI::obtainTargetDir()
{
  auto target_dir = _external_api->obtainTargetDir();
  if (target_dir == "") {
    target_dir = ".";
  }
  return target_dir;
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
  return _external_api->isSTAStarted();
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

// ugly: special case for timing
void PLAPI::reportPLInfo()
{
  LOG_INFO << "-----------------Start iPL Report Generation-----------------";

  ieda::Stats report_status;

  std::string output_dir = this->obtainTargetDir() + "/pl/report/";

  // std::string design_name = PlacerDBInst.get_design()->get_design_name();
  // std::string output_dir = "./evaluation_task/benchmark/" + design_name + "/pl_reports/";

  std::string summary_file = "summary_report.txt";
  std::ofstream summary_stream;
  summary_stream.open(output_dir + summary_file);
  if (!summary_stream.good()) {
    LOG_WARNING << "Cannot open file for summary report !";
  }
  summary_stream << "Generate the report at " << ieda::Time::getNowWallTime() << std::endl;

  // report base info
  reportPLBaseInfo(summary_stream);

  // report violation info
  reportViolationInfo(summary_stream);

  // report wirelength info
  reportWLInfo(summary_stream);

  // report density info
  reportBinDensity(summary_stream);

  // report timing info
  if (PlacerDBInst.get_placer_config()->isTimingEffort()) {
    reportTimingInfo(summary_stream);
  }

  // report congestion
  // if (PlacerDBInst.get_placer_config()->isCongestionEffort()) {
  //   reportCongestionInfo(summary_stream);
  // }
  summary_stream.close();

  double time_delta = report_status.elapsedRunTime();

  LOG_INFO << "Report Generation Total Time Elapsed: " << time_delta << "s";
  LOG_INFO << "-----------------Finish Report Generation-----------------";
}

void PLAPI::reportTopoInfo()
{
  _reporter->reportTopoInfo();
}

void PLAPI::reportWLInfo(std::ofstream& feed)
{
  _reporter->reportWLInfo(feed, this->obtainTargetDir() + "/pl/report");
}

void PLAPI::reportSTWLInfo(std::ofstream& feed)
{
  _reporter->reportSTWLInfo(feed);
}

void PLAPI::reportHPWLInfo(std::ofstream& feed)
{
  _reporter->reportHPWLInfo(feed);
}

void PLAPI::reportLongNetInfo(std::ofstream& feed)
{
  _reporter->reportLongNetInfo(feed);
}

void PLAPI::reportViolationInfo(std::ofstream& feed)
{
  _reporter->reportViolationInfo(feed, this->obtainTargetDir() + "/pl/report");
}

void PLAPI::reportBinDensity(std::ofstream& feed)
{
  _reporter->reportBinDensity(feed);
}

int32_t PLAPI::reportOverlapInfo(std::ofstream& feed)
{
  return _reporter->reportOverlapInfo(feed);
}

void PLAPI::reportLayoutWhiteInfo()
{
  _reporter->reportLayoutWhiteInfo(this->obtainTargetDir() + "/pl/report");
}

void PLAPI::reportTimingInfo(std::ofstream& feed)
{
  if (this->isSTAStarted()) {
    // this->initTimingEval();
    this->updateTiming(PlacerDBInst.get_topo_manager());
    _reporter->reportTimingInfo(feed);
  }
}

void PLAPI::reportCongestionInfo(std::ofstream& feed)
{
  _reporter->reportCongestionInfo(feed);
}

void PLAPI::reportPLBaseInfo(std::ofstream& feed)
{
  _reporter->reportPLBaseInfo(feed);
}

void PLAPI::printHPWLInfo()
{
  _reporter->printHPWLInfo();
}

void PLAPI::printTimingInfo()
{
  if (this->isSTAStarted()) {
    this->updateTiming(PlacerDBInst.get_topo_manager());
    _reporter->printTimingInfo();
  }
}

void PLAPI::saveNetPinInfoForDebug(std::string path)
{
  _reporter->saveNetPinInfoForDebug(path);
}

void PLAPI::savePinListInfoForDebug(std::string path)
{
  _reporter->savePinListInfoForDebug(path);
}

void PLAPI::plotConnectionForDebug(std::vector<std::string> net_name_list, std::string path)
{
  _reporter->plotConnectionForDebug(net_name_list, path);
}

void PLAPI::plotModuleListForDebug(std::vector<std::string> module_prefix_list, std::string path)
{
  _reporter->plotModuleListForDebug(module_prefix_list, path);
}

void PLAPI::plotModuleStateForDebug(std::vector<std::string> special_inst_list, std::string path)
{
  _reporter->plotModuleStateForDebug(special_inst_list, path);
}

ieval::TimingPin* wrapTimingTruePin(Node* node)
{
  ieval::TimingPin* timing_pin = new ieval::TimingPin();
  timing_pin->pin_name = node->get_name();
  timing_pin->x = node->get_location().get_x();
  timing_pin->y = node->get_location().get_y();
  timing_pin->is_real_pin = true;

  return timing_pin;
}

ieval::TimingPin* wrapTimingFakePin(int id, Point<int32_t> coordi)
{
  ieval::TimingPin* timing_pin = new ieval::TimingPin();
  timing_pin->pin_name = "fake_" + std::to_string(id);
  timing_pin->pin_id = id;
  timing_pin->x = coordi.get_x();
  timing_pin->y = coordi.get_y();
  timing_pin->is_real_pin = false;

  return timing_pin;
}

ieda_feature::PlaceSummary PLAPI::outputSummary(std::string step)
{
  ieda_feature::PlaceSummary summary;

  // 1:全局布局、详细布局、合法化都需要存储的数据参数，需要根据step存储不同的值
  auto place_density = PlacerDBInst.place_density;
  auto hpwl = PlacerDBInst.PL_HPWL;
  auto stwl = PlacerDBInst.PL_STWL;

  // 2:全局布局、详细布局需要存储的数据参数
  if (step == "place") {
    summary.gplace.place_density = place_density[0];
    summary.gplace.HPWL = hpwl[0];
    summary.gplace.STWL = stwl[0];

    summary.dplace.place_density = place_density[2];
    summary.dplace.HPWL = hpwl[2];
    summary.dplace.STWL = stwl[2];

    auto* pl_design = PlacerDBInst.get_design();
    summary.instance_cnt = pl_design->get_instances_range();
    int fix_inst_cnt = 0;
    for (auto* inst : pl_design->get_instance_list()) {
      if (inst->isFixed()) {
        fix_inst_cnt++;
      }
    }

    summary.fix_inst_cnt = fix_inst_cnt;
    summary.net_cnt = pl_design->get_nets_range();
    summary.total_pins = pl_design->get_pins_range();

    summary.bin_number = PlacerDBInst.get_placer_config()->get_nes_config().get_bin_cnt_x()
                         * PlacerDBInst.get_placer_config()->get_nes_config().get_bin_cnt_y();
    summary.bin_size_x = PlacerDBInst.bin_size_x;
    summary.bin_size_y = PlacerDBInst.bin_size_y;
    summary.overflow_number = PlacerDBInst.gp_overflow_number;
    summary.overflow = PlacerDBInst.gp_overflow;
  }
  // 3:合法化需要存储的数据参数
  else if (step == "legalization") {
    summary.lg_summary.pl_common_summary.HPWL = hpwl[1];
    summary.lg_summary.pl_common_summary.STWL = stwl[1];

    summary.lg_summary.lg_total_movement = PlacerDBInst.lg_total_movement;
    summary.lg_summary.lg_max_movement = PlacerDBInst.lg_max_movement;
  }

  return summary;
}

// private
PLAPI* PLAPI::_s_ipl_api_instance = nullptr;

}  // namespace ipl

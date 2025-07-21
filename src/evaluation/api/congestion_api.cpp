/*
 * @FilePath: congestion_api.cpp
 * @Author: Yihang Qiu (qiuyihang23@mails.ucas.ac.cn)
 * @Date: 2024-08-24 15:37:27
 * @Description:
 */

#include "congestion_api.h"

#include "congestion_eval.h"

namespace ieval {

#define EVAL_CONGESTION_INST (ieval::CongestionEval::getInst())

CongestionAPI* CongestionAPI::_congestion_api_inst = nullptr;

CongestionAPI::CongestionAPI()
{
}

CongestionAPI::~CongestionAPI()
{
}

CongestionAPI* CongestionAPI::getInst()
{
  if (_congestion_api_inst == nullptr) {
    _congestion_api_inst = new CongestionAPI();
  }

  return _congestion_api_inst;
}

void CongestionAPI::destroyInst()
{
  if (_congestion_api_inst != nullptr) {
    delete _congestion_api_inst;
    _congestion_api_inst = nullptr;
  }
}

EGRMapSummary CongestionAPI::egrMap(std::string stage)
{
  return egrMap(stage, EVAL_CONGESTION_INST->getEGRDirPath());
}

EGRMapSummary CongestionAPI::egrMapPure(std::string stage)
{
  return egrMapPure(stage, EVAL_CONGESTION_INST->getEGRDirPath());
}

EGRMapSummary CongestionAPI::egrMap(std::string stage, std::string rt_dir_path)
{
  EGRMapSummary egr_map_summary;

  EVAL_CONGESTION_INST->initEGR();
  egr_map_summary.horizontal_sum = EVAL_CONGESTION_INST->evalHoriEGR(stage, rt_dir_path);
  egr_map_summary.vertical_sum = EVAL_CONGESTION_INST->evalVertiEGR(stage, rt_dir_path);
  egr_map_summary.union_sum = EVAL_CONGESTION_INST->evalUnionEGR(stage, rt_dir_path);
  EVAL_CONGESTION_INST->destroyEGR();

  return egr_map_summary;
}

EGRMapSummary CongestionAPI::egrMapPure(std::string stage, std::string rt_dir_path)
{
  EGRMapSummary egr_map_summary;

  egr_map_summary.horizontal_sum = EVAL_CONGESTION_INST->evalHoriEGR(stage, rt_dir_path);
  egr_map_summary.vertical_sum = EVAL_CONGESTION_INST->evalVertiEGR(stage, rt_dir_path);
  egr_map_summary.union_sum = EVAL_CONGESTION_INST->evalUnionEGR(stage, rt_dir_path);

  return egr_map_summary;
}

RUDYMapSummary CongestionAPI::rudyMap(std::string stage, int32_t grid_size)
{
  RUDYMapSummary rudy_map_summary;

  EVAL_CONGESTION_INST->initIDB();
  rudy_map_summary = rudyMap(stage, EVAL_CONGESTION_INST->getCongestionNets(), EVAL_CONGESTION_INST->getCongestionRegion(),
                             grid_size * EVAL_CONGESTION_INST->getRowHeight());
  EVAL_CONGESTION_INST->destroyIDB();

  return rudy_map_summary;
}

RUDYMapSummary CongestionAPI::rudyMapPure(std::string stage, int32_t grid_size)
{
  RUDYMapSummary rudy_map_summary;
  rudy_map_summary = rudyMap(stage, EVAL_CONGESTION_INST->getCongestionNets(), EVAL_CONGESTION_INST->getCongestionRegion(),
                             grid_size * EVAL_CONGESTION_INST->getRowHeight());
  return rudy_map_summary;
}

RUDYMapSummary CongestionAPI::rudyMap(std::string stage, CongestionNets nets, CongestionRegion region, int32_t grid_size)
{
  RUDYMapSummary rudy_map_summary;

  rudy_map_summary.rudy_horizontal = EVAL_CONGESTION_INST->evalHoriRUDY(stage, nets, region, grid_size);
  rudy_map_summary.rudy_vertical = EVAL_CONGESTION_INST->evalVertiRUDY(stage, nets, region, grid_size);
  rudy_map_summary.rudy_union = EVAL_CONGESTION_INST->evalUnionRUDY(stage, nets, region, grid_size);

  rudy_map_summary.lutrudy_horizontal = EVAL_CONGESTION_INST->evalHoriLUTRUDY(stage, nets, region, grid_size);
  rudy_map_summary.lutrudy_vertical = EVAL_CONGESTION_INST->evalVertiLUTRUDY(stage, nets, region, grid_size);
  rudy_map_summary.lutrudy_union = EVAL_CONGESTION_INST->evalUnionLUTRUDY(stage, nets, region, grid_size);

  return rudy_map_summary;
}

OverflowSummary CongestionAPI::egrOverflow(std::string stage)
{
  return egrOverflow(stage, EVAL_CONGESTION_INST->getEGRDirPath());
}

OverflowSummary CongestionAPI::egrOverflow(std::string stage, std::string rt_dir_path)
{
  OverflowSummary overflow_summary;

  overflow_summary.total_overflow_horizontal = EVAL_CONGESTION_INST->evalHoriTotalOverflow(stage, rt_dir_path);
  overflow_summary.total_overflow_vertical = EVAL_CONGESTION_INST->evalVertiTotalOverflow(stage, rt_dir_path);
  overflow_summary.total_overflow_union = EVAL_CONGESTION_INST->evalUnionTotalOverflow(stage, rt_dir_path);

  overflow_summary.max_overflow_horizontal = EVAL_CONGESTION_INST->evalHoriMaxOverflow(stage, rt_dir_path);
  overflow_summary.max_overflow_vertical = EVAL_CONGESTION_INST->evalVertiMaxOverflow(stage, rt_dir_path);
  overflow_summary.max_overflow_union = EVAL_CONGESTION_INST->evalUnionMaxOverflow(stage, rt_dir_path);

  overflow_summary.weighted_average_overflow_horizontal = EVAL_CONGESTION_INST->evalHoriAvgOverflow(stage, rt_dir_path);
  overflow_summary.weighted_average_overflow_vertical = EVAL_CONGESTION_INST->evalVertiAvgOverflow(stage, rt_dir_path);
  overflow_summary.weighted_average_overflow_union = EVAL_CONGESTION_INST->evalUnionAvgOverflow(stage, rt_dir_path);

  return overflow_summary;
}

UtilizationSummary CongestionAPI::rudyUtilization(std::string stage, bool use_lut)
{
  UtilizationSummary utilization_summary;

  std::string rudy_dir_path = EVAL_CONGESTION_INST->getDefaultOutputDir() + "/RUDY_map";
  utilization_summary = rudyUtilization(stage, rudy_dir_path, use_lut);

  return utilization_summary;
}

UtilizationSummary CongestionAPI::rudyUtilization(std::string stage, std::string rudy_dir_path, bool use_lut)
{
  UtilizationSummary utilization_summary;

  utilization_summary.max_utilization_horizontal = EVAL_CONGESTION_INST->evalHoriMaxUtilization(stage, rudy_dir_path, use_lut);
  utilization_summary.max_utilization_vertical = EVAL_CONGESTION_INST->evalVertiMaxUtilization(stage, rudy_dir_path, use_lut);
  utilization_summary.max_utilization_union = EVAL_CONGESTION_INST->evalUnionMaxUtilization(stage, rudy_dir_path, use_lut);

  utilization_summary.weighted_average_utilization_horizontal = EVAL_CONGESTION_INST->evalHoriAvgUtilization(stage, rudy_dir_path, use_lut);
  utilization_summary.weighted_average_utilization_vertical = EVAL_CONGESTION_INST->evalVertiAvgUtilization(stage, rudy_dir_path, use_lut);
  utilization_summary.weighted_average_utilization_union = EVAL_CONGESTION_INST->evalUnionAvgUtilization(stage, rudy_dir_path, use_lut);

  return utilization_summary;
}

void CongestionAPI::evalNetInfo()
{
  EVAL_CONGESTION_INST->initIDB();
  EVAL_CONGESTION_INST->evalNetInfo();
  EVAL_CONGESTION_INST->destroyIDB();
}

void CongestionAPI::evalNetInfoPure()
{
  EVAL_CONGESTION_INST->evalNetInfo();
}

int CongestionAPI::findPinNumber(std::string net_name)
{
  return EVAL_CONGESTION_INST->findPinNumber(net_name);
}

int CongestionAPI::findAspectRatio(std::string net_name)
{
  return EVAL_CONGESTION_INST->findAspectRatio(net_name);
}

float CongestionAPI::findLness(std::string net_name)
{
  return EVAL_CONGESTION_INST->findLness(net_name);
}

int32_t CongestionAPI::findBBoxWidth(std::string net_name)
{
  return EVAL_CONGESTION_INST->findBBoxWidth(net_name);
}

int32_t CongestionAPI::findBBoxHeight(std::string net_name)
{
  return EVAL_CONGESTION_INST->findBBoxHeight(net_name);
}

int64_t CongestionAPI::findBBoxArea(std::string net_name)
{
  return EVAL_CONGESTION_INST->findBBoxArea(net_name);
}

int32_t CongestionAPI::findBBoxLx(std::string net_name)
{
  return EVAL_CONGESTION_INST->findBBoxLx(net_name);
}

int32_t CongestionAPI::findBBoxLy(std::string net_name)
{
  return EVAL_CONGESTION_INST->findBBoxLy(net_name);
}

int32_t CongestionAPI::findBBoxUx(std::string net_name)
{
  return EVAL_CONGESTION_INST->findBBoxUx(net_name);
}

int32_t CongestionAPI::findBBoxUy(std::string net_name)
{
  return EVAL_CONGESTION_INST->findBBoxUy(net_name);
}

std::string CongestionAPI::egrUnionMap(std::string stage, std::string rt_dir_path)
{
  EVAL_CONGESTION_INST->setEGRDirPath(rt_dir_path + "/rt/rt_temp_directory");
  EVAL_CONGESTION_INST->initEGR();
  std::string union_egr_map_path = EVAL_CONGESTION_INST->evalUnionEGR(stage, rt_dir_path + "/rt/rt_temp_directory");
  EVAL_CONGESTION_INST->destroyEGR();

  return union_egr_map_path;
}

std::map<std::string, std::vector<std::vector<int>>> CongestionAPI::getEGRMap(bool is_run_egr)
{
  return EVAL_CONGESTION_INST->getDemandSupplyDiffMap(is_run_egr);
}


std::map<int, double> CongestionAPI::patchRUDYCongestion(std::map<int, std::pair<std::pair<int, int>, std::pair<int, int>>> patch_coords)
{
  std::map<int, double> patch_rudy_congestion;

  EVAL_CONGESTION_INST->initIDB();

  CongestionEval congestion_eval;
  patch_rudy_congestion = congestion_eval.patchRUDYCongestion(EVAL_CONGESTION_INST->getCongestionNets(), 
                                                             patch_coords);
  EVAL_CONGESTION_INST->destroyIDB();

  return patch_rudy_congestion;
}

std::map<int, double> CongestionAPI::patchEGRCongestion(std::map<int, std::pair<std::pair<int, int>, std::pair<int, int>>> patch_coords)
{
  std::map<int, double> patch_egr_congestion;

  CongestionEval congestion_eval;
  patch_egr_congestion = congestion_eval.patchEGRCongestion(patch_coords);

  return patch_egr_congestion;
}

std::map<int, std::map<std::string, double>> CongestionAPI::patchLayerEGRCongestion(std::map<int, std::pair<std::pair<int, int>, std::pair<int, int>>> patch_coords)
{
  std::map<int, std::map<std::string, double>> patch_layer_egr_congestion;

  CongestionEval congestion_eval;
  patch_layer_egr_congestion = congestion_eval.patchLayerEGRCongestion(patch_coords);

  return patch_layer_egr_congestion;
}


}  // namespace ieval

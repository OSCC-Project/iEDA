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

EGRMapSummary CongestionAPI::egrMap()
{
  EGRMapSummary egr_map_summary;

  std::string rt_dir_path = "./rt_temp_directory";
  egr_map_summary = egrMap(rt_dir_path);

  return egr_map_summary;
}

EGRMapSummary CongestionAPI::egrMap(std::string rt_dir_path)
{
  EGRMapSummary egr_map_summary;

  EVAL_CONGESTION_INST->initEGR();
  egr_map_summary.horizontal_sum = EVAL_CONGESTION_INST->evalHoriEGR(rt_dir_path);
  egr_map_summary.vertical_sum = EVAL_CONGESTION_INST->evalVertiEGR(rt_dir_path);
  egr_map_summary.union_sum = EVAL_CONGESTION_INST->evalUnionEGR(rt_dir_path);
  EVAL_CONGESTION_INST->destroyEGR();

  return egr_map_summary;
}

RUDYMapSummary CongestionAPI::rudyMap(int32_t grid_size)
{
  RUDYMapSummary rudy_map_summary;

  EVAL_CONGESTION_INST->initIDB();
  rudy_map_summary = rudyMap(EVAL_CONGESTION_INST->getCongestionNets(), EVAL_CONGESTION_INST->getCongestionRegion(),
                             grid_size * EVAL_CONGESTION_INST->getRowHeight());
  EVAL_CONGESTION_INST->destroyIDB();

  return rudy_map_summary;
}

RUDYMapSummary CongestionAPI::rudyMap(CongestionNets nets, CongestionRegion region, int32_t grid_size)
{
  RUDYMapSummary rudy_map_summary;

  rudy_map_summary.rudy_horizontal = EVAL_CONGESTION_INST->evalHoriRUDY(nets, region, grid_size);
  rudy_map_summary.rudy_vertical = EVAL_CONGESTION_INST->evalVertiRUDY(nets, region, grid_size);
  rudy_map_summary.rudy_union = EVAL_CONGESTION_INST->evalUnionRUDY(nets, region, grid_size);

  rudy_map_summary.lutrudy_horizontal = EVAL_CONGESTION_INST->evalHoriLUTRUDY(nets, region, grid_size);
  rudy_map_summary.lutrudy_vertical = EVAL_CONGESTION_INST->evalVertiLUTRUDY(nets, region, grid_size);
  rudy_map_summary.lutrudy_union = EVAL_CONGESTION_INST->evalUnionLUTRUDY(nets, region, grid_size);

  return rudy_map_summary;
}

OverflowSummary CongestionAPI::egrOverflow()
{
  OverflowSummary overflow_summary;

  std::string rt_dir_path = "./rt_temp_directory";
  overflow_summary = egrOverflow(rt_dir_path);

  return overflow_summary;
}

OverflowSummary CongestionAPI::egrOverflow(std::string rt_dir_path)
{
  OverflowSummary overflow_summary;

  overflow_summary.total_overflow_horizontal = EVAL_CONGESTION_INST->evalHoriTotalOverflow(rt_dir_path);
  overflow_summary.total_overflow_vertical = EVAL_CONGESTION_INST->evalVertiTotalOverflow(rt_dir_path);
  overflow_summary.total_overflow_union = EVAL_CONGESTION_INST->evalUnionTotalOverflow(rt_dir_path);

  overflow_summary.max_overflow_horizontal = EVAL_CONGESTION_INST->evalHoriMaxOverflow(rt_dir_path);
  overflow_summary.max_overflow_vertical = EVAL_CONGESTION_INST->evalVertiMaxOverflow(rt_dir_path);
  overflow_summary.max_overflow_union = EVAL_CONGESTION_INST->evalUnionMaxOverflow(rt_dir_path);

  overflow_summary.weighted_average_overflow_horizontal = EVAL_CONGESTION_INST->evalHoriAvgOverflow(rt_dir_path);
  overflow_summary.weighted_average_overflow_vertical = EVAL_CONGESTION_INST->evalVertiAvgOverflow(rt_dir_path);
  overflow_summary.weighted_average_overflow_union = EVAL_CONGESTION_INST->evalUnionAvgOverflow(rt_dir_path);

  return overflow_summary;
}

UtilizationSummary CongestionAPI::rudyUtilization(bool use_lut)
{
  UtilizationSummary utilization_summary;

  std::string rudy_dir_path = "./RUDY_map";
  utilization_summary = rudyUtilization(rudy_dir_path, use_lut);

  return utilization_summary;
}

UtilizationSummary CongestionAPI::rudyUtilization(std::string rudy_dir_path, bool use_lut)
{
  UtilizationSummary utilization_summary;

  utilization_summary.max_utilization_horizontal = EVAL_CONGESTION_INST->evalHoriMaxUtilization(rudy_dir_path, use_lut);
  utilization_summary.max_utilization_vertical = EVAL_CONGESTION_INST->evalVertiMaxUtilization(rudy_dir_path, use_lut);
  utilization_summary.max_utilization_union = EVAL_CONGESTION_INST->evalUnionMaxUtilization(rudy_dir_path, use_lut);

  utilization_summary.weighted_average_utilization_horizontal = EVAL_CONGESTION_INST->evalHoriAvgUtilization(rudy_dir_path, use_lut);
  utilization_summary.weighted_average_utilization_vertical = EVAL_CONGESTION_INST->evalVertiAvgUtilization(rudy_dir_path, use_lut);
  utilization_summary.weighted_average_utilization_union = EVAL_CONGESTION_INST->evalUnionAvgUtilization(rudy_dir_path, use_lut);

  return utilization_summary;
}

void CongestionAPI::evalNetInfo()
{
  EVAL_CONGESTION_INST->initIDB();
  EVAL_CONGESTION_INST->evalNetInfo();
  EVAL_CONGESTION_INST->destroyIDB();
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

}  // namespace ieval

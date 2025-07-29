/*
 * @FilePath: feature_eval_union.cpp
 * @Author: Yihang Qiu (qiuyihang23@mails.ucas.ac.cn)
 * @Date: 2024-10-11 11:00:07
 * @Description:
 */

#include <algorithm>
#include <fstream>
#include <iostream>

#include "congestion_api.h"
#include "density_api.h"
#include "feature_builder.h"
#include "idm.h"
#include "timing_api.hh"
#include "union_api.h"
#include "wirelength_api.h"

namespace ieda_feature {

UnionEvalSummary FeatureBuilder::buildUnionEvalSummary(int32_t grid_size, std::string stage)
{
  UnionEvalSummary union_eval_summary;

  TotalWLSummary total_wl_summary;
  ieval::TotalWLSummary eval_total_wl_summary = WIRELENGTH_API_INST->totalWLPure(); 
  total_wl_summary.HPWL = eval_total_wl_summary.HPWL;
  total_wl_summary.FLUTE = eval_total_wl_summary.FLUTE;
  total_wl_summary.HTree = eval_total_wl_summary.HTree;
  total_wl_summary.VTree = eval_total_wl_summary.VTree;
  total_wl_summary.GRWL = eval_total_wl_summary.GRWL;

  DensityMapSummary density_map_summary;
  ieval::DensityMapSummary eval_density_map_summary = DENSITY_API_INST->densityMapPure(stage, grid_size);
  density_map_summary.cell_map_summary.macro_density = eval_density_map_summary.cell_map_summary.macro_density;
  density_map_summary.cell_map_summary.stdcell_density = eval_density_map_summary.cell_map_summary.stdcell_density;
  density_map_summary.cell_map_summary.allcell_density = eval_density_map_summary.cell_map_summary.allcell_density;
  density_map_summary.pin_map_summary.macro_pin_density = eval_density_map_summary.pin_map_summary.macro_pin_density;
  density_map_summary.pin_map_summary.stdcell_pin_density = eval_density_map_summary.pin_map_summary.stdcell_pin_density;
  density_map_summary.pin_map_summary.allcell_pin_density = eval_density_map_summary.pin_map_summary.allcell_pin_density;
  density_map_summary.net_map_summary.local_net_density = eval_density_map_summary.net_map_summary.local_net_density;
  density_map_summary.net_map_summary.global_net_density = eval_density_map_summary.net_map_summary.global_net_density;
  density_map_summary.net_map_summary.allnet_density = eval_density_map_summary.net_map_summary.allnet_density;
  if (stage == "place") {
    ieval::MacroMarginSummary eval_macro_margin_summary = DENSITY_API_INST->macroMarginMap(grid_size);
    density_map_summary.macro_margin_summary.horizontal_margin = eval_macro_margin_summary.horizontal_margin;
    density_map_summary.macro_margin_summary.vertical_margin = eval_macro_margin_summary.vertical_margin;
    density_map_summary.macro_margin_summary.union_margin = eval_macro_margin_summary.union_margin;
  }

  CongestionSummary congestion_summary;
  ieval::EGRMapSummary eval_egr_map_summary = CONGESTION_API_INST->egrMapPure(stage);
  congestion_summary.egr_map_summary.horizontal_sum = eval_egr_map_summary.horizontal_sum;
  congestion_summary.egr_map_summary.vertical_sum = eval_egr_map_summary.vertical_sum;
  congestion_summary.egr_map_summary.union_sum = eval_egr_map_summary.union_sum;
  ieval::RUDYMapSummary eval_rudy_map_summary = CONGESTION_API_INST->rudyMapPure(stage, grid_size);
  congestion_summary.rudy_map_summary.rudy_horizontal = eval_rudy_map_summary.rudy_horizontal;
  congestion_summary.rudy_map_summary.rudy_vertical = eval_rudy_map_summary.rudy_vertical;
  congestion_summary.rudy_map_summary.rudy_union = eval_rudy_map_summary.rudy_union;
  congestion_summary.rudy_map_summary.lutrudy_horizontal = eval_rudy_map_summary.lutrudy_horizontal;
  congestion_summary.rudy_map_summary.lutrudy_vertical = eval_rudy_map_summary.lutrudy_vertical;
  congestion_summary.rudy_map_summary.lutrudy_union = eval_rudy_map_summary.lutrudy_union;
  ieval::OverflowSummary eval_overflow_summary = CONGESTION_API_INST->egrOverflow(stage);
  congestion_summary.overflow_summary.total_overflow_horizontal = eval_overflow_summary.total_overflow_horizontal;
  congestion_summary.overflow_summary.total_overflow_vertical = eval_overflow_summary.total_overflow_vertical;
  congestion_summary.overflow_summary.total_overflow_union = eval_overflow_summary.total_overflow_union;
  congestion_summary.overflow_summary.max_overflow_horizontal = eval_overflow_summary.max_overflow_horizontal;
  congestion_summary.overflow_summary.max_overflow_vertical = eval_overflow_summary.max_overflow_vertical;
  congestion_summary.overflow_summary.max_overflow_union = eval_overflow_summary.max_overflow_union;
  congestion_summary.overflow_summary.weighted_average_overflow_horizontal = eval_overflow_summary.weighted_average_overflow_horizontal;
  congestion_summary.overflow_summary.weighted_average_overflow_vertical = eval_overflow_summary.weighted_average_overflow_vertical;
  congestion_summary.overflow_summary.weighted_average_overflow_union = eval_overflow_summary.weighted_average_overflow_union;
  ieval::UtilizationSummary eval_utilization_summary = CONGESTION_API_INST->rudyUtilization(stage, false);
  congestion_summary.rudy_utilization_summary.max_utilization_horizontal = eval_utilization_summary.max_utilization_horizontal;
  congestion_summary.rudy_utilization_summary.max_utilization_vertical = eval_utilization_summary.max_utilization_vertical;
  congestion_summary.rudy_utilization_summary.max_utilization_union = eval_utilization_summary.max_utilization_union;
  congestion_summary.rudy_utilization_summary.weighted_average_utilization_horizontal
      = eval_utilization_summary.weighted_average_utilization_horizontal;
  congestion_summary.rudy_utilization_summary.weighted_average_utilization_vertical
      = eval_utilization_summary.weighted_average_utilization_vertical;
  congestion_summary.rudy_utilization_summary.weighted_average_utilization_union
      = eval_utilization_summary.weighted_average_utilization_union;
  ieval::UtilizationSummary eval_lut_utilization_summary = CONGESTION_API_INST->rudyUtilization(stage, true);
  congestion_summary.lutrudy_utilization_summary.max_utilization_horizontal = eval_lut_utilization_summary.max_utilization_horizontal;
  congestion_summary.lutrudy_utilization_summary.max_utilization_vertical = eval_lut_utilization_summary.max_utilization_vertical;
  congestion_summary.lutrudy_utilization_summary.max_utilization_union = eval_lut_utilization_summary.max_utilization_union;
  congestion_summary.lutrudy_utilization_summary.weighted_average_utilization_horizontal
      = eval_lut_utilization_summary.weighted_average_utilization_horizontal;
  congestion_summary.lutrudy_utilization_summary.weighted_average_utilization_vertical
      = eval_lut_utilization_summary.weighted_average_utilization_vertical;
  congestion_summary.lutrudy_utilization_summary.weighted_average_utilization_union
      = eval_lut_utilization_summary.weighted_average_utilization_union;

  union_eval_summary.total_wl_summary = total_wl_summary;
  union_eval_summary.density_map_summary = density_map_summary;
  union_eval_summary.congestion_summary = congestion_summary;

  return union_eval_summary;
}

bool FeatureBuilder::initEvalTool()
{
  UNION_API_INST->initIDB();
  UNION_API_INST->initEGR(false);
  UNION_API_INST->initFlute();

  return true;
}

bool FeatureBuilder::destroyEvalTool()
{
  UNION_API_INST->destroyIDB();
  UNION_API_INST->destroyEGR();
  UNION_API_INST->destroyFlute();
  UNION_API_INST->destroyInst();

  return true;
}

bool FeatureBuilder::buildNetEval(std::string csv_path)
{
  std::cout << "Building net evaluation data...\n";

  auto* idb_builder = dmInst->get_idb_builder();
  idb::IdbDesign* idb_design = idb_builder->get_def_service()->get_design();

  CONGESTION_API_INST->evalNetInfoPure();
  WIRELENGTH_API_INST->evalNetInfoPure();
  // auto net_power_data = ieval::TimingAPI::getInst()->evalNetPower();

  std::ofstream csv_file(csv_path, std::ios::out | std::ios::binary);
  if (!csv_file) {
    return false;
  }

  csv_file
      << "net_name,net_type,pin_num,aspect_ratio,bbox_width,bbox_height,bbox_area,lx,ly,ux,uy,lness,hpwl,rsmt,grwl,hpwl_power,flute_"
         "power,egr_power,x_entropy,y_entropy,x_avg_nn_dist,x_std_nn_dist,x_ratio_nn_dist,y_avg_nn_dist,y_std_nn_dist,y_ratio_nn_dist\n";

  const size_t buffer_size = 1024 * 1024;  // 1MB buffer
  std::vector<char> buffer(buffer_size);
  csv_file.rdbuf()->pubsetbuf(buffer.data(), buffer_size);

  std::ostringstream oss;
  oss.str().reserve(buffer_size);  // Pre-allocate memory

  for (size_t i = 0; i < idb_design->get_net_list()->get_net_list().size(); i++) {
    auto* idb_net = idb_design->get_net_list()->get_net_list()[i];
    std::string net_name = idb_net->get_net_name();
    std::string net_type;

    int pin_num = CONGESTION_API_INST->findPinNumber(net_name);
    if (pin_num < 3) {
      continue;
    }
    int aspect_ratio = CONGESTION_API_INST->findAspectRatio(net_name);
    float l_ness = CONGESTION_API_INST->findLness(net_name);
    int32_t bbox_width = CONGESTION_API_INST->findBBoxWidth(net_name);
    int32_t bbox_height = CONGESTION_API_INST->findBBoxHeight(net_name);
    int64_t bbox_area = CONGESTION_API_INST->findBBoxArea(net_name);
    int32_t bbox_lx = CONGESTION_API_INST->findBBoxLx(net_name);
    int32_t bbox_ly = CONGESTION_API_INST->findBBoxLy(net_name);
    int32_t bbox_ux = CONGESTION_API_INST->findBBoxUx(net_name);
    int32_t bbox_uy = CONGESTION_API_INST->findBBoxUy(net_name);

    int32_t hpwl = WIRELENGTH_API_INST->findNetHPWL(net_name);
    int32_t flute = WIRELENGTH_API_INST->findNetFLUTE(net_name);
    int32_t grwl = WIRELENGTH_API_INST->findNetGRWL(net_name);

    double x_entropy = CONGESTION_API_INST->findXEntropy(net_name);
    double y_entropy = CONGESTION_API_INST->findYEntropy(net_name);
    double avg_x_nn_dist = CONGESTION_API_INST->findAvgXNNDistance(net_name);
    double std_x_nn_dist = CONGESTION_API_INST->findStdXNNDistance(net_name);
    double ratio_x_nn_dist = CONGESTION_API_INST->findRatioXNNDistance(net_name);
    double avg_y_nn_dist = CONGESTION_API_INST->findAvgYNNDistance(net_name);
    double std_y_nn_dist = CONGESTION_API_INST->findStdYNNDistance(net_name);
    double ratio_y_nn_dist = CONGESTION_API_INST->findRatioYNNDistance(net_name);

    // Remove backslashes in net_name to match timing && power evaluation data
    net_name.erase(std::remove(net_name.begin(), net_name.end(), '\\'), net_name.end());

    // if (ieval::TimingAPI::getInst()->isClockNet(net_name)) {
    //   net_type = "clock";
    // } else {
    //   net_type = "signal";
    // }
    net_type = "signal";

    // if (net_power_data["HPWL"].find(net_name) == net_power_data["HPWL"].end()
    //     || net_power_data["FLUTE"].find(net_name) == net_power_data["FLUTE"].end()
    //     || net_power_data["EGR"].find(net_name) == net_power_data["EGR"].end()) {
    //   std::cerr << "Error: net_name '" << net_name << "' not found in net_power_data.\n";
    //   std::exit(EXIT_FAILURE);
    // }

    // double hpwl_power = net_power_data["HPWL"][net_name];
    // double flute_power = net_power_data["FLUTE"][net_name];
    // double egr_power = net_power_data["EGR"][net_name];

    double hpwl_power = 0.0;
    double flute_power = 0.0;
    double egr_power = 0.0;

    csv_file << net_name << ',' << net_type << "," << pin_num << ',' << aspect_ratio << ',' << bbox_width << "," << bbox_height << ","
             << bbox_area << "," << bbox_lx << "," << bbox_ly << "," << bbox_ux << "," << bbox_uy << "," << l_ness << ',' << hpwl << ','
             << flute << ',' << grwl << ',' << hpwl_power << ',' << flute_power << "," << egr_power << "," << x_entropy << ',' << y_entropy
             << ',' << avg_x_nn_dist << "," << std_x_nn_dist << "," << ratio_x_nn_dist << "," << avg_y_nn_dist << "," << std_y_nn_dist
             << "," << ratio_y_nn_dist << '\n';

    if (oss.tellp() >= buffer_size / 2) {
      csv_file << oss.str();
      oss.str("");
      oss.clear();
    }
  }
  if (oss.tellp() > 0) {
    csv_file << oss.str();
  }
  csv_file.close();
  return true;
}

}  // namespace ieda_feature
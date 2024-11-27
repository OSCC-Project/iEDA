#include <iostream>

#include "congestion_api.h"
#include "idm.h"
#include "timing_api.hh"
#include "union_api.h"
#include "wirelength_api.h"

void TestBuildNetEval();

int main()
{
  TestBuildNetEval();

  return 0;
}

void TestBuildNetEval()
{
  dmInst->init("/data/project_share/dataset_baseline/jpeg_encoder/workspace/config/iEDA_config/db_default_config_tmp.json");
  UNION_API_INST->initIDB();
  UNION_API_INST->initEGR(true);
  UNION_API_INST->initFlute();

  std::cout << "Building net evaluation data...\n";

  auto* idb_builder = dmInst->get_idb_builder();
  idb::IdbDesign* idb_design = idb_builder->get_def_service()->get_design();

  CONGESTION_API_INST->evalNetInfoPure();
  WIRELENGTH_API_INST->evalNetInfoPure();
  ieval::TimingAPI::getInst()->evalTiming("EGR", true);
  ieval::TimingAPI::getInst()->evalTiming("HPWL");
  ieval::TimingAPI::getInst()->evalTiming("FLUTE");
  auto net_power_data = ieval::TimingAPI::getInst()->evalNetPower();

  for (size_t i = 0; i < idb_design->get_net_list()->get_net_list().size(); i++) {
    auto* idb_net = idb_design->get_net_list()->get_net_list()[i];
    std::string net_name = idb_net->get_net_name();
    std::string net_type;


    int pin_num = CONGESTION_API_INST->findPinNumber(net_name);
    if (pin_num < 4) {
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

    net_name.erase(std::remove(net_name.begin(), net_name.end(), '\\'), net_name.end());

    if (ieval::TimingAPI::getInst()->isClockNet(net_name)) {
      net_type = "clock";
    } else {
      net_type = "signal";
    }
    if (net_power_data["HPWL"].find(net_name) == net_power_data["HPWL"].end()
        || net_power_data["FLUTE"].find(net_name) == net_power_data["FLUTE"].end()
        || net_power_data["EGR"].find(net_name) == net_power_data["EGR"].end()) {
      std::cerr << "Error: net_name '" << net_name << "' not found in net_power_data.\n";
      std::exit(EXIT_FAILURE);
    }

    double hpwl_power = net_power_data["HPWL"][net_name];
    double flute_power = net_power_data["FLUTE"][net_name];
    double egr_power = net_power_data["EGR"][net_name];

    if (net_name == "InvalidSymbol") {
      std::cout << net_name << ',' << net_type << "," << pin_num << ',' << aspect_ratio << ',' << bbox_width << "," << bbox_height << ","
                << bbox_area << "," << bbox_lx << "," << bbox_ly << "," << bbox_ux << "," << bbox_uy << "," << l_ness << ',' << hpwl << ','
                << flute << ',' << grwl << ',' << hpwl_power << ',' << flute_power << "," << egr_power << '\n';
    }
  }
}
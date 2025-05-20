/*
 * @FilePath: wirelength_app.cpp
 * @Author: Yihang Qiu (qiuyihang23@mails.ucas.ac.cn)
 * @Date: 2024-08-24 15:37:27
 * @Description:
 */

#include <iostream>

#include "idm.h"
#include "wirelength_api.h"

void TestTotalWirelength();
void TestNetWirelength();
void TestPathWirelength();
void TestEgrWirelength(std::string guide_path);
void TestWirelengthEvalFromIDB(const std::string& db_config_path);

void PrintUsage(const char* program_name) {
  std::cout << "Wirelength Evaluation" << std::endl;
  std::cout << "Usage: " << program_name << " <function_name>" << std::endl;
  std::cout << "Available parameters:" << std::endl;
  std::cout << "  <db_config_path> Path to the database configuration file." << std::endl;
  std::cout << "  --help, -h       Show this help message and exit." << std::endl;
}

int main(const int argc, const char* argv[])
{
  if (argc == 2) {
    if (const std::string arg = argv[1]; arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return 0;
    } else {
      std::cout << "db_config_path: " << arg << std::endl;
      TestWirelengthEvalFromIDB(arg);
      // Here are some test functions that can be uncommented to run
      // TestTotalWirelength();
      // TestNetWirelength();
      // TestPathWirelength();
      // TestEgrWirelength("./rt_temp_directory/early_router/route.guide");
      return 0;
    }
  }

  return 0;
}

void TestTotalWirelength()
{
  ieval::WirelengthAPI wirelength_api;

  std::vector<std::pair<int32_t, int32_t>> point_set;
  std::pair<int32_t, int32_t> point1(0, 0);
  std::pair<int32_t, int32_t> point2(3, 6);
  std::pair<int32_t, int32_t> point3(4, 4);
  std::pair<int32_t, int32_t> point4(6, 3);

  point_set.push_back(point1);
  point_set.push_back(point2);
  point_set.push_back(point3);
  point_set.push_back(point4);

  std::vector<std::vector<std::pair<int32_t, int32_t>>> point_sets;
  point_sets.push_back(point_set);

  ieval::TotalWLSummary total_wl = wirelength_api.totalWL(point_sets);

  std::cout << "Total HPWL: " << total_wl.HPWL << std::endl;
  std::cout << "Total FLUTE: " << total_wl.FLUTE << std::endl;
  std::cout << "Total HTree: " << total_wl.HTree << std::endl;
  std::cout << "Total VTree: " << total_wl.VTree << std::endl;
}

void TestNetWirelength()
{
  ieval::WirelengthAPI wirelength_api;

  std::vector<std::pair<int32_t, int32_t>> point_set;
  std::pair<int32_t, int32_t> point1(0, 0);
  std::pair<int32_t, int32_t> point2(3, 6);
  std::pair<int32_t, int32_t> point3(4, 4);
  std::pair<int32_t, int32_t> point4(6, 3);

  point_set.push_back(point1);
  point_set.push_back(point2);
  point_set.push_back(point3);
  point_set.push_back(point4);

  ieval::NetWLSummary net_wl = wirelength_api.netWL(point_set);

  std::cout << "Net HPWL: " << net_wl.HPWL << std::endl;
  std::cout << "Net FLUTE: " << net_wl.FLUTE << std::endl;
  std::cout << "Net HTree: " << net_wl.HTree << std::endl;
  std::cout << "Net VTree: " << net_wl.VTree << std::endl;
}

void TestPathWirelength()
{
  ieval::WirelengthAPI wirelength_api;

  std::vector<std::pair<int32_t, int32_t>> point_set;
  std::pair<int32_t, int32_t> point1(0, 0);
  std::pair<int32_t, int32_t> point2(3, 6);
  std::pair<int32_t, int32_t> point3(4, 4);
  std::pair<int32_t, int32_t> point4(6, 3);

  point_set.push_back(point1);
  point_set.push_back(point2);
  point_set.push_back(point3);
  point_set.push_back(point4);

  std::pair<int32_t, int32_t> point_pair1(4, 4);
  std::pair<int32_t, int32_t> point_pair2(6, 3);

  ieval::PathWLSummary path_wl = wirelength_api.pathWL(point_set, {point_pair1, point_pair2});

  std::cout << "Path HPWL: " << path_wl.HPWL << std::endl;
  std::cout << "Path FLUTE: " << path_wl.FLUTE << std::endl;
  std::cout << "Path HTree: " << path_wl.HTree << std::endl;
  std::cout << "Path VTree: " << path_wl.VTree << std::endl;
}

void TestEgrWirelength(std::string guide_path)
{
  ieval::WirelengthAPI wirelength_api;

  float total_egr_wl = wirelength_api.totalEGRWL(guide_path);
  std::cout << "Total EGR WL: " << total_egr_wl << std::endl;

  float net_egr_wl = wirelength_api.netEGRWL(guide_path, "clk");
  std::cout << "Net EGR WL: " << net_egr_wl << std::endl;

  float path_egr_wl = wirelength_api.pathEGRWL(guide_path, "clk", "clk_0_buf:I");
  std::cout << "Path EGR WL: " << path_egr_wl << std::endl;
}

void TestWirelengthEvalFromIDB(const std::string& db_config_path)
{
  dmInst->init(db_config_path);
  ieval::WirelengthAPI wirelength_api;
  ieval::TotalWLSummary wl_summary = wirelength_api.totalWL();
  std::cout << "Total HPWL: " << wl_summary.HPWL << std::endl;
  // std::cout << "Total FLUTE: " << wl_summary.FLUTE << std::endl;
  // std::cout << "Total HTree: " << wl_summary.HTree << std::endl;
  // std::cout << "Total VTree: " << wl_summary.VTree << std::endl;
  // std::cout << "Total GRWL: " << wl_summary.GRWL << std::endl;

  // ieval::NetWLSummary net_wl_summary = wirelength_api.netWL("req_msg[31]");
  // std::cout << ">> Net: req_msg[31], Wirelength: " << std::endl;
  // std::cout << "Net HPWL: " << net_wl_summary.HPWL << std::endl;
  // std::cout << "Net FLUTE: " << net_wl_summary.FLUTE << std::endl;
  // std::cout << "Net HTree: " << net_wl_summary.HTree << std::endl;
  // std::cout << "Net VTree: " << net_wl_summary.VTree << std::endl;
  // std::cout << "Net GRWL: " << net_wl_summary.GRWL << std::endl;
}

/**
 * @File Name: report_manage.cpp
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2022-11-07
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "report_manager.h"

#include "idm.h"
#include "report_basic.h"
#include "report_db.h"
#include "report_drc.h"
#include "report_evaluator.h"
#include "report_place.h"
#include "report_route.h"

namespace iplf {

ReportManager* ReportManager::_instance = nullptr;
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool ReportManager::reportDBSummary(const std::string& file_name)
{
  ReportOStream os(file_name);
  ReportDB report_db("Report DB Summary");
  report_db.add_table(report_db.createSummaryTable());
  report_db.add_table(report_db.createSummaryInstances());
  report_db.add_table(report_db.createSummaryNets());
  report_db.add_table(report_db.createSummaryLayers());
  report_db.add_table(report_db.createSummaryPins());

  os << report_db;
  return true;
}

ReportOStream::ReportOStream(const std::string& file)
{
  if (not file.empty()) {
    _fs.open(file, std::ios_base::out | std::ios_base::trunc);
  }
}

ReportOStream::~ReportOStream()
{
  if (_fs.is_open()) {
    _fs.close();
  }
}

bool ReportManager::reportWL(const std::string& file_name)
{
  auto start = std::chrono::steady_clock::now();

  ReportOStream ofs(file_name);
  ReportEvaluator report("WireLengthReport");
  report.add_table(report.createWireLengthReport());

  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
  std::cout << "report time cost: " << ms << " ms" << std::endl;
  ofs << report;

  return true;
}

bool ReportManager::reportCongestion(const std::string& file_name)
{
  auto start = std::chrono::steady_clock::now();

  ReportOStream ofs(file_name);
  ReportEvaluator report("CongestionReport");
  report.add_table(report.createCongestionReport());

  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
  std::cout << "report time cost: " << ms << " ms" << std::endl;
  ofs << report;
  return true;
}

bool ReportManager::reportInstance(const std::string& file_name, const std::string& inst_name)
{
  ReportOStream ofs(file_name);
  ReportDesign report("InstanceReport");

  auto* inst = dmInst->get_idb_design()->get_instance_list()->find_instance(inst_name);

  if (!inst) {
    std::cout << "Can not find instance \"" << inst_name << "\"\n";
    return false;
  }
  report.add_table(report.createInstanceTable(inst_name));
  report.add_table(report.createInstancePinTable(inst));
  ofs << report;
  return true;
}

bool ReportManager::reportNet(const std::string& file_name, const std::string& net_name)
{
  ReportOStream ofs(file_name);
  ReportDesign report("Net Report");
  auto* net = dmInst->get_idb_design()->get_net_list()->find_net(net_name);
  if (!net) {
    std::cout << "Can not find net \"" << net_name << "\"\n";
    return false;
  }
  report.add_table(report.createNetTable(net));
  ofs << report;
  return true;
}

bool ReportManager::reportDanglingNet(const std::string& file_name)
{
  ReportOStream ofs(file_name);
  ReportDanglingNet report;
  ofs << report;
  std::cout << "Total Dangling Nets: " << report.get_count() << std::endl;
  if (ofs.fileOpen()) {
    std::cout << "Details outputs to " << file_name << std::endl;
  }
  return true;
}

bool ReportManager::reportRoute(const std::string& file, const std::string& net_name, bool summary)
{
  ReportOStream ofs(file);
  ReportRoute report("Report Route");
  // single net report
  if (!net_name.empty()) {
    auto* net = dmInst->get_idb_design()->get_net_list()->find_net(net_name);
    if (!net) {
      std::cout << "Cannot find net " << net_name << std::endl;
      return false;
    }
    report.createNetReport(net);
  }
  // summary report
  else if (summary) {
    report.createSummaryReport();
  }

  ofs << report;
  return true;
}

bool ReportManager::reportPlaceDistribution(const std::vector<std::string>& prefixes, const std::string& file)
{
  ReportPlace report("Distribution");
  report.createInstDistributionReport(prefixes, file);
  return true;
}

bool ReportManager::reportInstLevel(const std::string& prefix, int level, int num_threshold)
{
  ReportPlace report("Leveled Instances");
  report.createInstLevelReport(prefix, level, num_threshold);
  return true;
}

bool ReportManager::reportDRC(const std::string& file)
{
  ReportDRC report("DRC");
  report.createDrcReport();
  ReportOStream{file} << report;
  return true;
}

bool ReportManager::reportDRC(const std::string& file_name, std::map<std::string, int>& drc_result,
                              std::tuple<bool, std::vector<std::string>, std::vector<std::string>, int>& connectivity_result)
{
  ReportOStream os(file_name);
  ReportDRC report_drc("Report DRC Summary");
  report_drc.add_table(report_drc.createDRCTable(drc_result));
  report_drc.add_table(report_drc.createConnectivityTable(connectivity_result));
  report_drc.add_table(report_drc.createConnectivityDetailTable(connectivity_result));

  os << report_drc;
  return true;
}

}  // namespace iplf

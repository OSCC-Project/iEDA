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
#include "Reporter.h"

#include "api/TimingEngine.hh"

using namespace std;

namespace ito {

Reporter* Reporter::_instance = nullptr;

Reporter* Reporter::get_instance()
{
  static std::mutex mt;
  if (_instance == nullptr) {
    std::lock_guard<std::mutex> lock(mt);
    if (_instance == nullptr) {
      _instance = new Reporter();
    }
  }
  return _instance;
}

void Reporter::destroy_instance()
{
  if (_instance != nullptr) {
    delete _instance;
    _instance = nullptr;
  }
}

/**
 * @brief report start or end time
 *
 * @param begin
 * true: start time.   false: end time.
 */
void Reporter::reportTime(bool begin)
{
  if (!_outfile.is_open()) {
    _outfile.open(_output_path, ios::app);
  }
  time_t timep;
  time(&timep);
  char tmp[256];
  strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S", localtime(&timep));

  if (begin) {
    _outfile << "\n\n======================= Program start time " << tmp << "=======================" << endl;
  } else {
    _outfile << "======================= Program end time " << tmp << "=======================" << endl;
  }
  _outfile.close();
}

/**
 * @brief record violation count
 *
 * @param before true: before fix design.
 * false: after fix design.
 */
void Reporter::reportDRVResult(int slew_violations, int cap_violations, bool before)
{
  if (!_outfile.is_open()) {
    _outfile.open(_output_path, ios::app);
  }

  if (before) {
    _outfile << "TO: Total find " << slew_violations << " nets with slew violations.\n";
    _outfile << "TO: Total find " << cap_violations << " nets with capacitance violations.\n";
  } else {
    _outfile << "\nAfter the" << _check_count << "-th repair. \n\tThere are still " << slew_violations << " nets with slew violations, and "
             << cap_violations << " nets with capacitance violations.\n";
    _check_count++;
  }
  _outfile.close();
};

void Reporter::reportNetInfo(ista::Net* net, double cap_load_allowed_max)
{
  if (!_outfile.is_open()) {
    _outfile.open(_output_path, ios::app);
  }
  _outfile << "Net name: " << net->get_name() << endl;
  auto driver = net->getDriver();
  ista::Pin* driver_pin = dynamic_cast<ista::Pin*>(driver);
  _outfile << "  Driver: " << endl;
  _outfile << "\tDrvr Inst: " << driver_pin->get_own_instance()->get_name()
           << "\t|cell master: " << driver_pin->get_own_instance()->get_inst_cell()->get_cell_name()
           << "\t|Driver Pin: " << driver_pin->getFullName() << "\t|cap:" << driver->cap(ista::AnalysisMode::kMax, ista::TransType::kRise)
           << "\t|Driver max cap: " << cap_load_allowed_max << endl;

  std::vector<ista::DesignObject*> loads = net->getLoads();
  _outfile << "  Load: " << endl;
  double capacitance = 0;
  for (auto load : loads) {
    if (load->isPin()) {
      ista::Pin* load_pin = dynamic_cast<ista::Pin*>(load);
      _outfile << "\tLoad Inst: " << load_pin->get_own_instance()->get_name()
               << "\t|cell master: " << load_pin->get_own_instance()->get_inst_cell()->get_cell_name()
               << "\t|Load Pin: " << load_pin->getFullName() << "\t|cap:" << load->cap(ista::AnalysisMode::kMax, ista::TransType::kRise)
               << endl;
      capacitance += load->cap(ista::AnalysisMode::kMax, ista::TransType::kRise);
    }
  }
  _outfile << "Total load cap: " << capacitance << endl;
  _outfile.close();
}

void Reporter::reportHoldResult(std::vector<double> timing_slacks_hold, std::vector<int> hold_vio_num, std::vector<int> insert_buf_num,
                                double slack, int insert_buf)
{
  if (!_outfile.is_open()) {
    _outfile.open(_output_path, ios::app);
  }
  _outfile << "\n#buf : number of inserted buffer\n";
  _outfile << "#vio : number of hold violation endpoints\n";
  _outfile << "------------------------------------------------------" << endl;
  _outfile << setiosflags(ios::left) << setw(18) << "#buf" << resetiosflags(ios::left) << setiosflags(ios::right) << setw(18) << "Hold WNS"
           << setw(18) << "#vio" << resetiosflags(ios::right) << endl;
  _outfile << "------------------------------------------------------" << endl;
  _outfile << setiosflags(ios::left) << setw(18) << 0 << resetiosflags(ios::left) << setiosflags(ios::right) << setw(18) << timing_slacks_hold[0]
           << setw(18) << hold_vio_num[0] << resetiosflags(ios::right) << endl;

  for (size_t i = 0; i < insert_buf_num.size(); ++i) {
    _outfile << setiosflags(ios::left) << setw(18) << insert_buf_num[i] << resetiosflags(ios::left) << setiosflags(ios::right) << setw(18)
             << timing_slacks_hold[i + 1] << setw(18) << hold_vio_num[i + 1] << resetiosflags(ios::right) << endl;
  }
  _outfile << "------------------------------------------------------" << endl;
  _outfile.close();
}

void Reporter::report(const std::string info)
{
  if (!_outfile.is_open()) {
    _outfile.open(_output_path, ios::app);
  }
  _outfile << info << endl;
  _outfile.close();
}

void Reporter::reportSetupResult(std::vector<double> slack_store)
{
  if (!_outfile.is_open()) {
    _outfile.open(_output_path, ios::app);
  }
  for (auto slack : slack_store) {
    _outfile << slack << " ";
  }
  _outfile << endl;
  _outfile.close();
}
}  // namespace ito

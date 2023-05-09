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

/**
 * @brief report start or end time
 *
 * @param begin
 * true: start time.   false: end time.
 */
void Reporter::reportTime(bool begin) {
  if (!_outfile.is_open()) {
    _outfile.open(_output_path, ios::app);
  }
  time_t timep;
  time(&timep);
  char tmp[256];
  strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S", localtime(&timep));

  if (begin) {
    _outfile << "\n\n======================= Program start time " << tmp
             << "=======================" << endl;
  } else {
    _outfile << "======================= Program end time " << tmp
             << "=======================" << endl;
  }
  _outfile.close();
}

/**
 * @brief record violation count
 *
 * @param before true: before fix design.
 * false: after fix design.
 */
void Reporter::reportDRVResult(int repair_count, int slew_violations,
                               int length_violations, int cap_violations,
                               int fanout_violations, bool before) {
  if (!_outfile.is_open()) {
    _outfile.open(_output_path, ios::app);
  }

  if (before) {
    _outfile << "Found " << slew_violations << " slew violations." << endl;
    _outfile << "Found " << cap_violations << " capacitance violations."
             << endl;
    _outfile << "Found " << fanout_violations << " fanout violations." << endl;
    _outfile << "Found " << length_violations << " long wires." << endl;
    _outfile << "Before ViolationFix | slew_vio: " << slew_violations
             << " cap_vio: " << cap_violations
             << " fanout_vio: " << fanout_violations
             << " length_vio: " << length_violations << endl;
  } else {
    _outfile << "The " << _check_count << "th check" << endl;
    _outfile << "After ViolationFix | slew_vio: " << slew_violations
             << " cap_vio: " << cap_violations
             << " fanout_vio: " << fanout_violations
             << " length_vio: " << length_violations << endl;
    _check_count++;
  }
  _outfile.close();
};

void Reporter::reportNetInfo(ista::Net *net, double max_cap) {
  if (!_outfile.is_open()) {
    _outfile.open(_output_path, ios::app);
  }
  _outfile << "Net name: " << net->get_name() << endl;
  auto drvr = net->getDriver();
  ista::Pin *drvr_pin = dynamic_cast<ista::Pin *>(drvr);
  _outfile << "  Drvr: " << endl;
  _outfile << "\tDrvr Inst: " << drvr_pin->get_own_instance()->get_name()
           << "\t|cell master: "
           << drvr_pin->get_own_instance()->get_inst_cell()->get_cell_name()
           << "\t|Drvr Pin: " << drvr_pin->getFullName() << "\t|cap:"
           << drvr->cap(ista::AnalysisMode::kMax, ista::TransType::kRise)
           << "\t|Drvr max cap: " << max_cap << endl;

  vector<ista::DesignObject *> loads = net->getLoads();
  _outfile << "  Load: " << endl;
  double capacitance = 0;
  for (auto load : loads) {
    if (load->isPin()) {
      ista::Pin *load_pin = dynamic_cast<ista::Pin *>(load);
      _outfile << "\tLoad Inst: " << load_pin->get_own_instance()->get_name()
               << "\t|cell master: "
               << load_pin->get_own_instance()->get_inst_cell()->get_cell_name()
               << "\t|Load Pin: " << load_pin->getFullName() << "\t|cap:"
               << load->cap(ista::AnalysisMode::kMax, ista::TransType::kRise)
               << endl;
      capacitance +=
          load->cap(ista::AnalysisMode::kMax, ista::TransType::kRise);
    }
  }
  _outfile << "Total load cap: " << capacitance << endl;
  _outfile.close();
}

void Reporter::reportHoldResult(vector<double> hold_slacks,
                                vector<int> hold_vio_num,
                                vector<int> insert_buf_num, double slack,
                                int insert_buf) {
  if (!_outfile.is_open()) {
    _outfile.open(_output_path, ios::app);
  }
  _outfile << "\n#buf : number of inserted buffer\n";
  _outfile << "#vio : number of hold violation endpoints\n";
  _outfile << "------------------------------------------------------" << endl;
  _outfile << setiosflags(ios::left) << setw(18) << "#buf"
           << resetiosflags(ios::left) << setiosflags(ios::right) << setw(18) << "Hold WNS"
           << setw(18) << "#vio" << resetiosflags(ios::right) << endl;
  _outfile << "------------------------------------------------------" << endl;
  _outfile << setiosflags(ios::left) << setw(18) << 0 << resetiosflags(ios::left)
           << setiosflags(ios::right) << setw(18) << hold_slacks[0] << setw(18)
           << hold_vio_num[0] << resetiosflags(ios::right) << endl;

  for (size_t i = 0; i < insert_buf_num.size(); ++i) {
    _outfile << setiosflags(ios::left) << setw(18) << insert_buf_num[i]
             << resetiosflags(ios::left) << setiosflags(ios::right) << setw(18)
             << hold_slacks[i + 1] << setw(18) << hold_vio_num[i + 1]
             << resetiosflags(ios::right) << endl;
  }
  _outfile << "------------------------------------------------------" << endl;
  _outfile.close();
}

void Reporter::report(const string info) {
  if (!_outfile.is_open()) {
    _outfile.open(_output_path, ios::app);
  }
  _outfile << info << endl;
  _outfile.close();
}

void Reporter::reportSetupResult(std::vector<double> slack_store) {
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

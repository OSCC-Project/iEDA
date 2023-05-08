#pragma once

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "ctime"
#include "ids.hpp"

using std::string;
using std::vector;

namespace ito {
using std::endl;
using std::ios;
using std::ofstream;
using std::setw;
using std::string;
using std::vector;

class Reporter {
 public:
  Reporter() = default;
  Reporter(const string path) : _output_path(path) {}
  ~Reporter() = default;

  void reportTime(bool begin);

  void reportDRVResult(int repair_count, int slew_violations, int length_violations,
                       int cap_violations, int fanout_violations, bool before);
  void reportSetupResult(std::vector<double> slack_store);
  void reportHoldResult(vector<double> hold_slacks, vector<int> hold_vio_num,
                        vector<int> insert_buf_num, double slack, int insert_buf);

  void reportNetInfo(ista::Net *net, double max_cap);

  void report(const string info);

  ofstream &get_ofstream() {
    if (!_outfile.is_open()) {
      _outfile.open(_output_path, std::ios::app);
    }
    return _outfile;
  }

 private:
  string   _output_path;
  ofstream _outfile;

  int _check_count = 1;
};
} // namespace ito

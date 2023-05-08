#pragma once

#include <fstream>
#include <iostream>
#include <string>

using std::string;

namespace ino {
using std::endl;
using std::ios;
using std::ofstream;
using std::string;

class Reporter {
 public:
  Reporter() = default;
  Reporter(string path) : _output_path(path) {}
  ~Reporter() = default;

  void reportTime(bool begin);

  void report(string info);

  ofstream &get_ofstream() {
    _outfile.open(_output_path, std::ios::app);
    return _outfile;
  }

 private:
  string   _output_path;
  ofstream _outfile;

  int _check_count = 1;
};
} // namespace ino

#pragma once

#include <fstream>
#include <iostream>
#include <string>

#include "ids.hpp"

using idb::IdbDesign;
using idb::IdbBlockage;
using idb::IdbInstance;

namespace ito {
using std::ofstream;
using std::cout;
using std::endl;

class GDSwriter {
 public:
  GDSwriter() = default;
  ~GDSwriter() = default;

  static void writeGDS(idb::IdbBuilder *idb_builder, std::string path);

  static void writeBlockage(ofstream &gds_file, idb::IdbBlockage *block, int layer, int named);

  static void writeAllInstance(ofstream &gds_file, idb::IdbDesign *idb_design);

  static void writeInstance(ofstream &gds_file, idb::IdbInstance *instance, int layer);
};
} // namespace ito

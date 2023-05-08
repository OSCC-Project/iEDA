#pragma once

#include <iostream>
#include <string>

#include "ids.hpp"

namespace ito {

class DbInterface;
class ToConfig;
class Tree;

class iTO {
 public:
  iTO() = delete;
  iTO(const std::string &config_file);
  ~iTO();
  iTO(const iTO &other) = delete;
  iTO(iTO &&other) = delete;

  DbInterface *get_db_interface() {
    // initialization();
    return _db_interface;
  }

  ToConfig *get_config() { return _to_config; }

  /// operator
  void initialization(idb::IdbBuilder *idb_build, ista::TimingEngine *timing);
  void resetInitialization(idb::IdbBuilder    *idb_build,
                           ista::TimingEngine *timing_engine = nullptr);

  void runTO();
  // void fixFanout();
  void optimizeDesignViolation();
  void optimizeSetup();
  void optimizeHold();

  std::vector<idb::IdbNet *> optimizeCTSDesignViolation(idb::IdbNet *idb_net, Tree *topo);

 private:
  DbInterface *_db_interface = nullptr;
  ToConfig    *_to_config = nullptr;
};

} // namespace ito

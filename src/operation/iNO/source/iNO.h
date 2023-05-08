#pragma once

#include "FixFanout.h"

#include <iostream>
#include <string>

namespace ino {

class NoConfig;
class DbInterface;

class iNO {
 public:
  iNO() = delete;
  iNO(const std::string &config_file);
  ~iNO();

  //   DbInterface *get_db_interface() { return _db_interface; }
  NoConfig *get_config() { return _no_config; }

  void fixFanout();

  void initialization(idb::IdbBuilder *idb_build, ista::TimingEngine *timing);
 private:

  // data
  DbInterface *_db_interface;
  NoConfig    *_no_config = nullptr;
};

} // namespace ino

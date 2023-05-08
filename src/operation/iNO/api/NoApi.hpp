#pragma once

#include <any>
#include <map>
#include <string>
#include <vector>

#include "ids.hpp"

namespace ino {

#define NoApiInst (ino::NoApi::getInst())

class NoApi {
 public:
  static NoApi &getInst();
  static void   destroyInst();

  void initNO(const std::string &ITO_CONFIG_PATH);
  void iNODataInit(idb::IdbBuilder *idb = nullptr, ista::TimingEngine *timing = nullptr);
  // void resetiTOData(idb::IdbBuilder *idb, ista::TimingEngine *timing = nullptr);
  // // function API
  void fixFanout();

  void saveDef(std::string saved_def_path = "");

  NoConfig *get_no_config();

  void reportTiming();

 private:
  static NoApi *_no_api_instance;
  NoApi() = default;
  NoApi(const NoApi &other) = delete;
  NoApi(NoApi &&other) = delete;
  ~NoApi() = default;
  NoApi &operator=(const NoApi &other) = delete;
  NoApi &operator=(NoApi &&other) = delete;

  idb::IdbBuilder    *initIDB();
  ista::TimingEngine *initISTA(idb::IdbBuilder *idb);

  ino::iNO           *_ino = nullptr;
  idb::IdbBuilder    *_idb = nullptr;
  ista::TimingEngine *_timing_engine = nullptr;
};

} // namespace ino

#pragma once

#include "DataManager.hpp"
#include "DetailedRouter.hpp"
#include "EarlyGlobalRouter.hpp"
#include "GDSPlotter.hpp"
#include "GlobalRouter.hpp"
#include "Monitor.hpp"
#include "PinAccessor.hpp"
#include "ResourceAllocator.hpp"
#include "TrackAssigner.hpp"
#include "ViolationRepairer.hpp"

namespace irt {

#define RT_INST (irt::RT::getInst())

class RT
{
 public:
  static void initInst(std::map<std::string, std::any>& config_map, idb::IdbBuilder* idb_builder);
  static RT& getInst();
  static void destroyInst();
  // function
  DataManager& getDataManager();

 private:
  static RT* _rt_instance;
  // config_map & idb_builder
  std::map<std::string, std::any> _config_map;
  idb::IdbBuilder* _idb_builder = nullptr;
  // config & database
  DataManager _data_manager;

  RT(std::map<std::string, std::any> config_map, idb::IdbBuilder* idb_builder)
  {
    _config_map = config_map;
    _idb_builder = idb_builder;
    init();
  }
  RT(const RT& other) = delete;
  RT(RT&& other) = delete;
  ~RT() { destroy(); }
  RT& operator=(const RT& other) = delete;
  RT& operator=(RT&& other) = delete;
  // function
  void init();
  void printHeader();
  void destroy();
  void printFooter();
};

}  // namespace irt

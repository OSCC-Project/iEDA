#pragma once

#include "Config.hpp"
#include "DRConfig.hpp"
#include "DRDatabase.hpp"
#include "DRNet.hpp"
#include "Database.hpp"

namespace irt {

class DRDataManager
{
 public:
  DRDataManager() = default;
  DRDataManager(const DRDataManager& other) = delete;
  DRDataManager(DRDataManager&& other) = delete;
  ~DRDataManager() = default;
  DRDataManager& operator=(const DRDataManager& other) = delete;
  DRDataManager& operator=(DRDataManager&& other) = delete;
  // function
  void input(Config& config, Database& database);
  DRConfig& getConfig() { return _dr_config; }
  DRDatabase& getDatabase() { return _dr_database; }
  std::vector<DRNet> convertToDRNetList(std::vector<Net>& net_list);
  DRNet convertToDRNet(Net& net);

 private:
  DRConfig _dr_config;
  DRDatabase _dr_database;
  // function
  void wrapConfig(Config& config);
  void wrapDatabase(Database& database);
  void wrapMicronDBU(Database& database);
  void wrapGCellAxis(Database& database);
  void wrapDie(Database& database);
  void wrapRoutingLayerList(Database& database);
  void wrapCutLayerList(Database& database);
  void wrapLayerViaMasterList(Database& database);
  void wrapRoutingBlockageList(Database& database);
  void buildConfig();
  void buildDatabase();
};

}  // namespace irt

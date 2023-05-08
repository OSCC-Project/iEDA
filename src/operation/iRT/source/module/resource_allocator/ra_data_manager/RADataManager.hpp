#pragma once

#include "Config.hpp"
#include "Database.hpp"
#include "RAConfig.hpp"
#include "RADatabase.hpp"
#include "RAModel.hpp"

namespace irt {

class RADataManager
{
 public:
  RADataManager() = default;
  RADataManager(const RADataManager& other) = delete;
  RADataManager(RADataManager&& other) = delete;
  ~RADataManager() = default;
  RADataManager& operator=(const RADataManager& other) = delete;
  RADataManager& operator=(RADataManager&& other) = delete;
  // function
  void input(Config& config, Database& database);
  RAConfig& getConfig() { return _ra_config; }
  RADatabase& getDatabase() { return _ra_database; }
  std::vector<RANet> convertToRANetList(std::vector<Net>& net_list);
  RANet convertToRANet(Net& net);

 private:
  RAConfig _ra_config;
  RADatabase _ra_database;
  // function
  void wrapConfig(Config& config);
  void wrapDatabase(Database& database);
  void wrapGCellAxis(Database& database);
  void wrapDie(Database& database);
  void wrapRoutingLayerList(Database& database);
  void wrapLayerViaMasterList(Database& database);
  void wrapRoutingBlockageList(Database& database);
  void buildConfig();
  void buildDatabase();
};

}  // namespace irt

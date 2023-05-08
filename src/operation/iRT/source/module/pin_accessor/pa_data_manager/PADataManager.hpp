#pragma once

#include "Config.hpp"
#include "Database.hpp"
#include "PAConfig.hpp"
#include "PADatabase.hpp"
#include "PANet.hpp"

namespace irt {

class PADataManager
{
 public:
  PADataManager() = default;
  PADataManager(const PADataManager& other) = delete;
  PADataManager(PADataManager&& other) = delete;
  ~PADataManager() = default;
  PADataManager& operator=(const PADataManager& other) = delete;
  PADataManager& operator=(PADataManager&& other) = delete;
  // function
  void input(Config& config, Database& database);
  PAConfig& getConfig() { return _pa_config; }
  PADatabase& getDatabase() { return _pa_database; }
  std::vector<PANet> convertToPANetList(std::vector<Net>& net_list);
  PANet convertToPANet(Net& net);

 private:
  PAConfig _pa_config;
  PADatabase _pa_database;
  // function
  void wrapConfig(Config& config);
  void wrapDatabase(Database& database);
  void wrapDie(Database& database);
  void wrapCellAxis(Database& database);
  void wrapRoutingLayerList(Database& database);
  void wrapLayerViaMasterList(Database& database);
  void wrapRoutingBlockageList(Database& database);
  void buildConfig();
  void buildDatabase();
};

}  // namespace irt

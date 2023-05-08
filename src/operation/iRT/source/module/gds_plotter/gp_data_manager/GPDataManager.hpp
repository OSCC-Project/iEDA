#pragma once

#include "Config.hpp"
#include "Database.hpp"
#include "GPConfig.hpp"
#include "GPDatabase.hpp"

namespace irt {

class GPDataManager
{
 public:
  GPDataManager() = default;
  GPDataManager(const GPDataManager& other) = delete;
  GPDataManager(GPDataManager&& other) = delete;
  ~GPDataManager() = default;
  GPDataManager& operator=(const GPDataManager& other) = delete;
  GPDataManager& operator=(GPDataManager&& other) = delete;
  // function
  void input(Config& config, Database& database);
  GPConfig& getConfig() { return _gp_config; }
  GPDatabase& getDatabase() { return _gp_database; }

 private:
  GPConfig _gp_config;
  GPDatabase _gp_database;
  // function
  void wrapConfig(Config& config);
  void wrapDatabase(Database& database);
  void wrapGCellAxis(Database& database);
  void wrapDie(Database& database);
  void wrapViaLib(Database& database);
  void wrapLayerList(Database& database);
  void wrapBlockageList(Database& database);
  void buildConfig();
  void buildDatabase();
  void buildGDSLayerMap();
  void buildLypFile();
};

}  // namespace irt

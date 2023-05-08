#pragma once

#include "Config.hpp"
#include "Database.hpp"
#include "GRConfig.hpp"
#include "GRDatabase.hpp"

namespace irt {

class GRDataManager
{
 public:
  GRDataManager() = default;
  GRDataManager(const GRDataManager& other) = delete;
  GRDataManager(GRDataManager&& other) = delete;
  ~GRDataManager() = default;
  GRDataManager& operator=(const GRDataManager& other) = delete;
  GRDataManager& operator=(GRDataManager&& other) = delete;
  // function
  void input(Config& config, Database& database);
  GRConfig& getConfig() { return _gr_config; }
  GRDatabase& getDatabase() { return _gr_database; }
  std::vector<GRNet> convertToGRNetList(std::vector<Net>& net_list);
  GRNet convertToGRNet(Net& net);

 private:
  GRConfig _gr_config;
  GRDatabase _gr_database;
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

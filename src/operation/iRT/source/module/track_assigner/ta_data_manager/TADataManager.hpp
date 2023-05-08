#pragma once

#include "Config.hpp"
#include "Database.hpp"
#include "Pin.hpp"
#include "TAConfig.hpp"
#include "TADatabase.hpp"
#include "TANet.hpp"

namespace irt {

class TADataManager
{
 public:
  TADataManager() = default;
  TADataManager(const TADataManager& other) = delete;
  TADataManager(TADataManager&& other) = delete;
  ~TADataManager() = default;
  TADataManager& operator=(const TADataManager& other) = delete;
  TADataManager& operator=(TADataManager&& other) = delete;
  // function
  void input(Config& config, Database& database);
  TAConfig& getConfig() { return _ta_config; }
  TADatabase& getDatabase() { return _ta_database; }
  std::vector<TANet> convertToTANetList(std::vector<Net>& net_list);
  TANet convertToTANet(Net& net);

 private:
  TAConfig _ta_config;
  TADatabase _ta_database;
  // function
  void wrapConfig(Config& config);
  void wrapDatabase(Database& database);
  void wrapMicronDBU(Database& database);
  void wrapGCellAxis(Database& database);
  void wrapDie(Database& database);
  void wrapRoutingLayerList(Database& database);
  void wrapLayerViaMasterList(Database& database);
  void wrapRoutingBlockageList(Database& database);
  void buildConfig();
  void buildDatabase();
};

}  // namespace irt

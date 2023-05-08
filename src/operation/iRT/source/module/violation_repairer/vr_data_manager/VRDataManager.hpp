#pragma once

#include "Config.hpp"
#include "Database.hpp"
#include "VRConfig.hpp"
#include "VRDatabase.hpp"
#include "VRNet.hpp"

namespace irt {

class VRDataManager
{
 public:
  VRDataManager() = default;
  VRDataManager(const VRDataManager& other) = delete;
  VRDataManager(VRDataManager&& other) = delete;
  ~VRDataManager() = default;
  VRDataManager& operator=(const VRDataManager& other) = delete;
  VRDataManager& operator=(VRDataManager&& other) = delete;
  // function
  void input(Config& config, Database& database);
  std::vector<VRNet> convertToVRNetList(std::vector<Net>& net_list);
  VRNet convertToVRNet(Net& net);
  VRConfig& getConfig() { return _vr_config; }
  VRDatabase& getDatabase() { return _vr_database; }

 private:
  VRConfig _vr_config;
  VRDatabase _vr_database;
  // function
  void wrapConfig(Config& config);
  void wrapDatabase(Database& database);
  void wrapMicronDBU(Database& database);
  void wrapGCellAxis(Database& database);
  void wrapViaMasterList(Database& database);
  void wrapRoutingLayerList(Database& database);
  void buildConfig();
  void buildDatabase();
};

}  // namespace irt

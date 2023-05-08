#ifndef IDRC_SRC_CFG_CONFIGURATOR_H_
#define IDRC_SRC_CFG_CONFIGURATOR_H_

#include <fstream>
#include <iostream>
#include <list>
#include <string>
#include <vector>

#include "DRCCOMUtil.h"
#include "DrcConfig.h"
#include "json.hpp"

namespace idrc {
class DrcConfigurator
{
 public:
  explicit DrcConfigurator() {}
  DrcConfigurator(const DrcConfigurator& other) = delete;
  DrcConfigurator(DrcConfigurator&& other) = delete;
  ~DrcConfigurator() {}
  DrcConfigurator& operator=(const DrcConfigurator& other) = delete;
  DrcConfigurator& operator=(DrcConfigurator&& other) = delete;
  // getter

  // setter

  // function
  void set(DrcConfig* config, std::string& drc_config_path);

 private:
  // function
  void initConfig(DrcConfig* config, std::string& dr_config_path);
  void initConfigByJson(DrcConfig* config, nlohmann::json& json);
  nlohmann::json getDataByJson(nlohmann::json value, std::vector<std::string> flag_list);
  void checkConfig(DrcConfig* config);
  void printConfig(DrcConfig* config);
};

}  // namespace idrc
#endif  // IDR_ACTION_CONFIGURATOR_H_
#include "DrcConfigurator.h"

namespace idrc {

void DrcConfigurator::set(DrcConfig* config, std::string& drc_config_path)
{
  initConfig(config, drc_config_path);
  // checkConfig(config);
  printConfig(config);
}

void DrcConfigurator::initConfig(DrcConfig* config, std::string& drc_config_path)
{
  std::ifstream drc_config_stream(drc_config_path);
  if (drc_config_stream) {
    nlohmann::json json;
    // read a JSON file
    drc_config_stream >> json;
    // std::cout << "[Configurator Info] The configuration file '" << drc_config_path << "' is read successfully!" << std::endl;
    initConfigByJson(config, json);
  } else {
    std::cout << "[Configurator Error] Failed to read configuration file '" << drc_config_path << "'!" << std::endl;
    exit(1);
  }
  drc_config_stream.close();
}

void DrcConfigurator::initConfigByJson(DrcConfig* config, nlohmann::json& json)
{
  // *************** INPUT ***************
  std::vector<std::string>& lef_paths = config->get_lef_paths();
  lef_paths.emplace_back(getDataByJson(json, {"INPUT", "tech_lef_path"}));
  for (std::string lef_path : getDataByJson(json, {"INPUT", "lef_paths"})) {
    lef_paths.emplace_back(lef_path);
  }
  config->set_def_path(getDataByJson(json, {"INPUT", "def_path"}));
  // ***************OUTPUT ***************
  // config->set_output_dir_path(getDataByJson(json, {"OUTPUT", "output_dir_path"}));
}

nlohmann::json DrcConfigurator::getDataByJson(nlohmann::json value, std::vector<std::string> ftag_list)
{
  int ftag_size = ftag_list.size();
  if (ftag_size == 0) {
    std::cout << "[Configurator Error] The number of json ftag is zero!" << std::endl;
    exit(1);
  }

  for (int i = 0; i < ftag_size; i++) {
    value = value[ftag_list[i]];
  }

  if (!value.is_null()) {
    return value;
  }
  std::cout << "[Configurator Error] The configuration file key=[";
  for (int i = 0; i < ftag_size; i++) {
    std::cout << ftag_list[i];
    if (i < ftag_size - 1) {
      std::cout << ".";
    }
  }
  std::cout << "] is null! exit..." << std::endl;
  exit(1);
}

void DrcConfigurator::checkConfig(DrcConfig* config)
{
}

void DrcConfigurator::printConfig(DrcConfig* config)
{
  return;
  // std::cout << "[Configurator Info] lef_paths:" << std::endl;
  // std::vector<std::string>& lef_paths = config->get_lef_paths();
  // for (size_t i = 0; i < lef_paths.size(); i++) {
  //   std::cout << "[Configurator Info]     " << lef_paths[i] << std::endl;
  // }
  // std::cout << "[Configurator Info] def_path: " << std::endl;
  // std::cout << "[Configurator Info]     " << config->get_def_path() << std::endl;

  // std::cout << "[Configurator Info] output_dir_path: " << std::endl;
  // std::cout << "[Configurator Info]     " << config->get_output_dir_path() << std::endl;
}

}  // namespace idrc
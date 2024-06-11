// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
#include "tcl_util.h"

namespace tcl {

// public

void TclUtil::addOption(TclCmd* tcl_ptr, std::vector<std::pair<std::string, ValueType>> config_list)
{
  for (auto [config_name, value_type] : config_list) {
    TclUtil::addOption(tcl_ptr, config_name, value_type);
  }
}

void TclUtil::addOption(TclCmd* tcl_ptr, std::string config_name, ValueType type)
{
  switch (type) {
    case ValueType::kInt:
      tcl_ptr->addOption(new TclIntOption(config_name.c_str(), 0));
      break;
    case ValueType::kIntList:
      tcl_ptr->addOption(new TclIntListOption(config_name.c_str(), 0));
      break;
    case ValueType::kDouble:
      tcl_ptr->addOption(new TclDoubleOption(config_name.c_str(), 0));
      break;
    case ValueType::kDoubleList:
      tcl_ptr->addOption(new TclDoubleListOption(config_name.c_str(), 0));
      break;
    case ValueType::kString:
      tcl_ptr->addOption(new TclStringOption(config_name.c_str(), 0));
      break;
    case ValueType::kStringList:
      tcl_ptr->addOption(new TclStringListOption(config_name.c_str(), 0));
      break;
    case ValueType::kStringDoubleMap:
      tcl_ptr->addOption(new TclStringListOption(config_name.c_str(), 0));
      break;
    default:
      std::cout << "[TclUtil] The value type is error!" << std::endl;
      exit(0);
      break;
  }
}

std::map<std::string, std::any> TclUtil::getConfigMap(TclCmd* tcl_ptr, std::vector<std::pair<std::string, ValueType>> config_list)
{
  std::map<std::string, std::any> config_map;
  for (auto [config_name, value_type] : config_list) {
    std::any config_value = TclUtil::getValue(tcl_ptr, config_name, value_type);
    if (config_value.has_value()) {
      config_map.insert(std::make_pair(config_name, config_value));
    }
  }
  return config_map;
}

std::any TclUtil::getValue(TclCmd* tcl_ptr, std::string config_name, ValueType type)
{
  std::any config_value;
  TclOption* option = tcl_ptr->getOptionOrArg(config_name.c_str());
  if (!option->is_set_val()) {
    return config_value;
  }
  switch (type) {
    case ValueType::kInt:
      config_value = option->getIntVal();
      break;
    case ValueType::kIntList:
      config_value = option->getIntList();
      break;
    case ValueType::kDouble:
      config_value = option->getDoubleVal();
      break;
    case ValueType::kDoubleList:
      config_value = option->getDoubleList();
      break;
    case ValueType::kString:
      config_value = std::string(option->getStringVal());
      break;
    case ValueType::kStringList:
      config_value = option->getStringList();
      break;
    case ValueType::kStringDoubleMap: {
      std::map<std::string, double> string_double_map;
      for (std::string temp : option->getStringList()) {
        std::vector<std::string> result_list = splitString(temp, ':');
        string_double_map.insert(std::make_pair(result_list[0], std::stof(result_list[1])));
      }
      config_value = string_double_map;
      break;
    }
    default:
      std::cout << "[TclUtil] The value type is error!" << std::endl;
      exit(0);
      break;
  }
  return config_value;
}

bool TclUtil::alterJsonConfig(std::string json_path, std::map<std::string, std::any> config_map)
{
  ordered_json config;
  if(json_path.empty()){
    LOG_ERROR << "Failed: There is no json_path to be found.";
    return 0;
  }

  // 从json_path先读入json
  std::ifstream file(json_path);
  if (!file.is_open()) {
    LOG_ERROR << "Failed to open JSON file for reading.";
    return 0;
  }
  file >> config;
  file.close();

  modifyJson(config, config_map);

  // for (auto [key, value] : config_map) {
  //   // Remove the first character '-' from the parameter list.
  //   std::string sub_key = key.substr(1, key.size() - 1);
  //   if (value.type() == typeid(std::string)) {
  //     config[sub_key] = std::any_cast<std::string>(value);
  //   } else if (value.type() == typeid(double)) {
  //     config[sub_key] = std::any_cast<double>(value);
  //   } else if (value.type() == typeid(int)) {
  //     config[sub_key] = std::any_cast<int>(value);
  //   } else if (value.type() == typeid(std::vector<int>)) {
  //     config[sub_key] = std::any_cast<std::vector<int>>(value);
  //   } else if (value.type() == typeid(std::vector<std::string>)) {
  //     config[sub_key] = std::any_cast<std::vector<std::string>>(value);
  //   }
  // }

  std::ofstream ofs(json_path);
  if (!ofs.is_open()) {
    LOG_ERROR << "Failed to open JSON file for writing.";
    return 0;
  }
  ofs << std::setw(4) << config;
  ofs.close();

  return 1;
  
}

}  // namespace tcl

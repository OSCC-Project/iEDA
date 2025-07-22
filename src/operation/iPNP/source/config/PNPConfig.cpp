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
/**
 * @file PNPConfig.cpp
 * @author Jianrong Su
 * @brief
 * @version 1.0
 * @date 2025-06-23
 */

#include "PNPConfig.hh"
#include "log/Log.hh"
#include <fstream>
#include "json.hpp"

namespace ipnp {

bool loadConfigFromJson(const std::string& config_file_path, PNPConfig* config) {
  if (!config) {
    LOG_ERROR << "PNPConfig pointer is null!" << std::endl;
    return false;
  }

  if (config_file_path.empty()) {
    LOG_WARNING << "Config file path is empty, using default configuration." << std::endl;
    return false;
  }

  try {
    std::ifstream file(config_file_path);
    if (!file.is_open()) {
      LOG_ERROR << "Failed to open config file: " << config_file_path << std::endl;
      return false;
    }

    nlohmann::json json_data;
    file >> json_data;

    if (json_data.contains("design")) {
      auto& design = json_data["design"];
      
      if (design.contains("lef_files") && design["lef_files"].is_array()) {
        std::vector<std::string> lef_files;
        for (const auto& lef : design["lef_files"]) {
          lef_files.push_back(lef.get<std::string>());
        }
        config->set_lef_files(lef_files);
      }

      if (design.contains("def_file") && design["def_file"].is_string()) {
        config->set_def_path(design["def_file"].get<std::string>());
      }

      if (design.contains("output_def_file") && design["output_def_file"].is_string()) {
        config->set_output_def_path(design["output_def_file"].get<std::string>());
      }
      
      if (design.contains("sdc_file") && design["sdc_file"].is_string()) {
        config->set_sdc_file(design["sdc_file"].get<std::string>());
      }
    }

    if (json_data.contains("lib")) {
      auto& lib = json_data["lib"];
      
      if (lib.contains("liberty_files") && lib["liberty_files"].is_array()) {
        std::vector<std::string> liberty_files;
        for (const auto& liberty : lib["liberty_files"]) {
          liberty_files.push_back(liberty.get<std::string>());
        }
        config->set_liberty_files(liberty_files);
      }
    }

    if (json_data.contains("timing")) {
      auto& timing = json_data["timing"];
      
      if (timing.contains("design_workspace") && timing["design_workspace"].is_string()) {
        config->set_timing_design_workspace(timing["design_workspace"].get<std::string>());
      }
    }

    if (json_data.contains("power")) {
      auto& power = json_data["power"];
      
      if (power.contains("power_net_name") && power["power_net_name"].is_string()) {
        config->set_power_net_name(power["power_net_name"].get<std::string>());
      }
    }

    if (json_data.contains("egr")) {
      auto& egr = json_data["egr"];

      if (egr.contains("map_path") && egr["map_path"].is_string()) {
        config->set_egr_map_path(egr["map_path"].get<std::string>());
      }
    }

    if (json_data.contains("grid")) {
      auto& grid = json_data["grid"];
      
      if (grid.contains("power_layers") && grid["power_layers"].is_array()) {
        std::vector<int> power_layers;
        for (const auto& layer : grid["power_layers"]) {
          power_layers.push_back(layer.get<int>());
        }
        config->set_power_layers(power_layers);
      }

      if (grid.contains("ho_region_num") && grid["ho_region_num"].is_number()) {
        config->set_ho_region_num(grid["ho_region_num"].get<int>());
      }

      if (grid.contains("ver_region_num") && grid["ver_region_num"].is_number()) {
        config->set_ver_region_num(grid["ver_region_num"].get<int>());
      }
    }

    // Load template configurations
    if (json_data.contains("templates")) {
      auto& templates = json_data["templates"];
      
      // Load horizontal templates
      if (templates.contains("horizontal") && templates["horizontal"].is_array()) {
        std::vector<TemplateConfig> horizontal_templates;
        for (const auto& template_data : templates["horizontal"]) {
          TemplateConfig template_config;
          template_config.direction = "horizontal";
          
          if (template_data.contains("width") && template_data["width"].is_number()) {
            template_config.width = template_data["width"].get<double>();
          }
          
          if (template_data.contains("pg_offset") && template_data["pg_offset"].is_number()) {
            template_config.pg_offset = template_data["pg_offset"].get<double>();
          }
          
          if (template_data.contains("space") && template_data["space"].is_number()) {
            template_config.space = template_data["space"].get<double>();
          }
          
          if (template_data.contains("offset") && template_data["offset"].is_number()) {
            template_config.offset = template_data["offset"].get<double>();
          }
          
          horizontal_templates.push_back(template_config);
        }
        config->set_horizontal_templates(horizontal_templates);
      }
      
      // Load vertical templates
      if (templates.contains("vertical") && templates["vertical"].is_array()) {
        std::vector<TemplateConfig> vertical_templates;
        for (const auto& template_data : templates["vertical"]) {
          TemplateConfig template_config;
          template_config.direction = "vertical";
          
          if (template_data.contains("width") && template_data["width"].is_number()) {
            template_config.width = template_data["width"].get<double>();
          }
          
          if (template_data.contains("pg_offset") && template_data["pg_offset"].is_number()) {
            template_config.pg_offset = template_data["pg_offset"].get<double>();
          }
          
          if (template_data.contains("space") && template_data["space"].is_number()) {
            template_config.space = template_data["space"].get<double>();
          }
          
          if (template_data.contains("offset") && template_data["offset"].is_number()) {
            template_config.offset = template_data["offset"].get<double>();
          }
          
          vertical_templates.push_back(template_config);
        }
        config->set_vertical_templates(vertical_templates);
      }
    }

    if (json_data.contains("simulated_annealing")) {
      auto& sa = json_data["simulated_annealing"];
      
      if (sa.contains("initial_temp") && sa["initial_temp"].is_number()) {
        config->set_sa_initial_temp(sa["initial_temp"].get<double>());
      }
      
      if (sa.contains("cooling_rate") && sa["cooling_rate"].is_number()) {
        config->set_sa_cooling_rate(sa["cooling_rate"].get<double>());
      }
      
      if (sa.contains("min_temp") && sa["min_temp"].is_number()) {
        config->set_sa_min_temp(sa["min_temp"].get<double>());
      }
      
      if (sa.contains("iterations_per_temp") && sa["iterations_per_temp"].is_number()) {
        config->set_sa_iterations_per_temp(sa["iterations_per_temp"].get<int>());
      }
      
      if (sa.contains("ir_drop_weight") && sa["ir_drop_weight"].is_number()) {
        config->set_sa_ir_drop_weight(sa["ir_drop_weight"].get<double>());
      }
      
      if (sa.contains("overflow_weight") && sa["overflow_weight"].is_number()) {
        config->set_sa_overflow_weight(sa["overflow_weight"].get<double>());
      }
      
      if (sa.contains("modifiable_layer_min") && sa["modifiable_layer_min"].is_number()) {
        config->set_sa_modifiable_layer_min(sa["modifiable_layer_min"].get<int>());
      }
      
      if (sa.contains("modifiable_layer_max") && sa["modifiable_layer_max"].is_number()) {
        config->set_sa_modifiable_layer_max(sa["modifiable_layer_max"].get<int>());
      }
    }

    LOG_INFO << "Successfully loaded configuration from " << config_file_path << std::endl;
    return true;
  } catch (const std::exception& e) {
    LOG_ERROR << "Error parsing JSON config file: " << e.what() << std::endl;
    return false;
  }
}

}  // namespace ipnp
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
 * @file PNPConfig.hh
 * @author Jianrong Su
 * @brief
 * @version 1.0
 * @date 2025-06-23
 */

#pragma once

#include <iostream>
#include <list>
#include <string>
#include <vector>

namespace ipnp {

  // Template configuration structure
  struct TemplateConfig {
    std::string direction;
    double width;        
    double pg_offset;    
    double space;        
    double offset;       
  };

  class PNPConfig
  {
  public:
    PNPConfig() = default;
    ~PNPConfig() = default;

    void set_lef_files(const std::vector<std::string>& lef_files) { _lef_files = lef_files; }
    const std::vector<std::string>& get_lef_files() const { return _lef_files; }

    void set_def_path(const std::string& def_path) { _def_path = def_path; }
    const std::string& get_def_path() const { return _def_path; }

    void set_output_def_path(const std::string& output_def_path) { _output_def_path = output_def_path; }
    const std::string& get_output_def_path() const { return _output_def_path; }

    void set_sdc_file(const std::string& sdc_file) { _sdc_file = sdc_file; }
    const std::string& get_sdc_file() const { return _sdc_file; }

    void set_power_layers(const std::vector<int>& power_layers) { _power_layers = power_layers; }
    const std::vector<int>& get_power_layers() const { return _power_layers; }

    void set_ho_region_num(int ho_region_num) { _ho_region_num = ho_region_num; }
    int get_ho_region_num() const { return _ho_region_num; }

    void set_ver_region_num(int ver_region_num) { _ver_region_num = ver_region_num; }
    int get_ver_region_num() const { return _ver_region_num; }

    void set_liberty_files(const std::vector<std::string>& liberty_files) { _liberty_files = liberty_files; }
    const std::vector<std::string>& get_liberty_files() const { return _liberty_files; }

    void set_timing_design_workspace(const std::string& workspace) { _timing_design_workspace = workspace; }
    const std::string& get_timing_design_workspace() const { return _timing_design_workspace; }

    void set_power_net_name(const std::string& name) { _power_net_name = name; }
    const std::string& get_power_net_name() const { return _power_net_name; }

    void set_egr_map_path(const std::string& egr_map_path) { _egr_map_path = egr_map_path; }
    const std::string& get_egr_map_path() const { return _egr_map_path; }

    void set_pl_default_config_path(const std::string& pl_default_config_path) { _pl_default_config_path = pl_default_config_path; }
    const std::string& get_pl_default_config_path() const { return _pl_default_config_path; }

    void set_report_path(const std::string& report_path) { _report_path = report_path; }
    const std::string& get_report_path() const { return _report_path; }

    void set_sa_initial_temp(double initial_temp) { _sa_initial_temp = initial_temp; }
    double get_sa_initial_temp() const { return _sa_initial_temp; }

    void set_sa_cooling_rate(double cooling_rate) { _sa_cooling_rate = cooling_rate; }
    double get_sa_cooling_rate() const { return _sa_cooling_rate; }

    void set_sa_min_temp(double min_temp) { _sa_min_temp = min_temp; }
    double get_sa_min_temp() const { return _sa_min_temp; }

    void set_sa_iterations_per_temp(int iterations) { _sa_iterations_per_temp = iterations; }
    int get_sa_iterations_per_temp() const { return _sa_iterations_per_temp; }

    void set_sa_ir_drop_weight(double weight) { _sa_ir_drop_weight = weight; }
    double get_sa_ir_drop_weight() const { return _sa_ir_drop_weight; }

    void set_sa_overflow_weight(double weight) { _sa_overflow_weight = weight; }
    double get_sa_overflow_weight() const { return _sa_overflow_weight; }

    // Range of modifiable layers for simulated annealing algorithm
    void set_sa_modifiable_layer_min(int layer) { _sa_modifiable_layer_min = layer; }
    int get_sa_modifiable_layer_min() const { return _sa_modifiable_layer_min; }

    void set_sa_modifiable_layer_max(int layer) { _sa_modifiable_layer_max = layer; }
    int get_sa_modifiable_layer_max() const { return _sa_modifiable_layer_max; }

    // Template configuration methods
    void set_horizontal_templates(const std::vector<TemplateConfig>& templates) { _horizontal_templates = templates; }
    const std::vector<TemplateConfig>& get_horizontal_templates() const { return _horizontal_templates; }

    void set_vertical_templates(const std::vector<TemplateConfig>& templates) { _vertical_templates = templates; }
    const std::vector<TemplateConfig>& get_vertical_templates() const { return _vertical_templates; }

  private:
    // design info
    std::vector<std::string> _lef_files;
    std::string _def_path;
    std::string _output_def_path;
    std::string _sdc_file;

    // lib info
    std::vector<std::string> _liberty_files;
    
    // grid info
    std::vector<int> _power_layers;
    int _ho_region_num;
    int _ver_region_num;
    
    std::string _timing_design_workspace;
    std::string _power_net_name;
    std::string _egr_map_path;
    std::string _pl_default_config_path;
    std::string _report_path;
    
    // sa info
    double _sa_initial_temp;
    double _sa_cooling_rate;
    double _sa_min_temp;
    int _sa_iterations_per_temp;
    double _sa_ir_drop_weight;
    double _sa_overflow_weight;
    int _sa_modifiable_layer_min;
    int _sa_modifiable_layer_max;

    // template configuration
    std::vector<TemplateConfig> _horizontal_templates;
    std::vector<TemplateConfig> _vertical_templates;
  };

  bool loadConfigFromJson(const std::string& config_file_path, PNPConfig* config);

}  // namespace ipnp

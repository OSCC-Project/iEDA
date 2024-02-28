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
#include <any>
#include <iomanip>

#include "tcl_plconfig.h"
#include "tcl_util.h"

namespace tcl {

CmdPLConfig::CmdPLConfig(const char* cmd_name) : TclCmd(cmd_name)
{
    // config_json_path string      required
    _config_list.push_back(std::make_pair("-config_json_path", ValueType::kString));
    // is_max_length_opt int
    _config_list.push_back(std::make_pair("-is_max_length_opt", ValueType::kInt));
    // max_length_constraint int
    _config_list.push_back(std::make_pair("-max_length_constraint", ValueType::kInt));
    // is_timing_aware_mode int
    _config_list.push_back(std::make_pair("-is_timing_aware_mode", ValueType::kInt));
    // ignore_net_degree int
    _config_list.push_back(std::make_pair("-ignore_net_degree", ValueType::kInt));
    // num_threads int
    _config_list.push_back(std::make_pair("-num_threads", ValueType::kInt));
    // init_wirelength_coef double
    _config_list.push_back(std::make_pair("-init_wirelength_coef", ValueType::kDouble));
    // referemce_hpwl int
    _config_list.push_back(std::make_pair("-reference_hpwl", ValueType::kInt));
    // min_wirelength_force_bar int
    _config_list.push_back(std::make_pair("-min_wirelength_force_bar", ValueType::kInt));
    // target_density double
    _config_list.push_back(std::make_pair("-target_density", ValueType::kDouble));
    // bin_cnt_x int
    _config_list.push_back(std::make_pair("-bin_cnt_x", ValueType::kInt));
    // bin_cnt_y int
    _config_list.push_back(std::make_pair("-bin_cnt_y", ValueType::kInt));
    // max_iter int
    _config_list.push_back(std::make_pair("-max_iter", ValueType::kInt));
    // max_backtrack int
    _config_list.push_back(std::make_pair("-max_backtrack", ValueType::kInt));
    // init_density_penalty double
    _config_list.push_back(std::make_pair("-init_density_penalty", ValueType::kDouble));
    // target_overflow double
    _config_list.push_back(std::make_pair("-target_overflow", ValueType::kDouble));
    // initial_prev_coordi_update_coef int
    _config_list.push_back(std::make_pair("-initial_prev_coordi_update_coef", ValueType::kInt));
    // min_precondition double
    _config_list.push_back(std::make_pair("-min_precondition", ValueType::kDouble));
    // min_phi_coef double
    _config_list.push_back(std::make_pair("-min_phi_coef", ValueType::kDouble));
    // max_phi_coef double
    _config_list.push_back(std::make_pair("-max_phi_coef", ValueType::kDouble));
    // max_buffer_num int
    _config_list.push_back(std::make_pair("-max_buffer_num", ValueType::kInt));
    // buffer_type stringlist
    _config_list.push_back(std::make_pair("-buffer_type", ValueType::kStringList));
    // max_displacement int
    _config_list.push_back(std::make_pair("-max_displacement", ValueType::kInt));
    // global_right_padding int
    _config_list.push_back(std::make_pair("-global_right_padding", ValueType::kInt));
    // max_displacement int
    _config_list.push_back(std::make_pair("-max_displacement", ValueType::kInt));
    // global_right_padding int
    _config_list.push_back(std::make_pair("-global_right_padding", ValueType::kInt));
    // enable_networkflow int
    _config_list.push_back(std::make_pair("-enable_networkflow", ValueType::kInt));
    // first_iter stringlist
    _config_list.push_back(std::make_pair("-first_iter", ValueType::kStringList));
    // second_iter stringlist
    _config_list.push_back(std::make_pair("-second_iter", ValueType::kStringList));
    // min_filler_width int
    _config_list.push_back(std::make_pair("-min_filler_width", ValueType::kInt));
    // fixed_macro stringlist
    _config_list.push_back(std::make_pair("-fixed_macro", ValueType::kStringList));
    // fixed_macro_coordinate stringlist
    _config_list.push_back(std::make_pair("-fixed_macro_coordinate", ValueType::kStringList));
    // blockage stringlist
    _config_list.push_back(std::make_pair("-blockage", ValueType::kStringList));
    // guidance_macro stringlist
    _config_list.push_back(std::make_pair("-guidance_macro", ValueType::kStringList));
    // guidance     stringlist
    _config_list.push_back(std::make_pair("-guidance", ValueType::kStringList));
    // solution_type string
    _config_list.push_back(std::make_pair("-solution_type", ValueType::kString));
    // perturb_per_step int
    _config_list.push_back(std::make_pair("-perturb_per_step", ValueType::kInt));
    // cool_rate double
    _config_list.push_back(std::make_pair("-cool_rate", ValueType::kDouble));
    // parts int
    _config_list.push_back(std::make_pair("-parts", ValueType::kInt));
    // ufactor int
    _config_list.push_back(std::make_pair("-ufactor", ValueType::kInt));
    // new_macro_density double
    _config_list.push_back(std::make_pair("-new_macro_density", ValueType::kDouble));
    // halo_x int
    _config_list.push_back(std::make_pair("-halo_x", ValueType::kInt));
    // halo_y int
    _config_list.push_back(std::make_pair("-halo_y", ValueType::kInt));
    // output_path string
    _config_list.push_back(std::make_pair("-output_path", ValueType::kString));

    TclUtil::addOption(this, _config_list);
}

unsigned CmdPLConfig::exec()
{
    std::map<std::string, std::any> config_map = TclUtil::getConfigMap(this, _config_list);
    
    if (config_map.empty()) {
        return 0;
    }
    std::string config_json_path = std::any_cast<std::string>(config_map["-config_json_path"]);
    config_map.erase("-config_json_path");
    TclUtil::alterJsonConfig(config_json_path, config_map);
    return 1;
}

}  // namespace tcl
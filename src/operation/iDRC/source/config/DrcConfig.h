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

#ifndef IDRC_SRC_CFG_DRC_CONFIG_H_
#define IDRC_SRC_CFG_DRC_CONFIG_H_

#include <map>
#include <string>
#include <vector>

namespace idrc {

class DrcConfig
{
 public:
  DrcConfig() {}
  ~DrcConfig() {}

  // getter
  std::vector<std::string>& get_lef_paths() { return _lef_paths; }
  std::string& get_def_path() { return _def_path; }
  // OUTPUT
  std::string& get_output_dir_path() { return _output_dir_path; }

  // setter
  // INPUT
  void set_lef_paths(const std::vector<std::string>& lef_paths) { _lef_paths = lef_paths; }
  void set_def_path(const std::string& def_path) { _def_path = def_path; }
  // OUTPUT
  void set_output_dir_path(const std::string& output_dir_path) { _output_dir_path = output_dir_path; }
  // function

 private:
  // INPUT
  std::vector<std::string> _lef_paths;
  std::string _def_path;
  // OUTPUT
  std::string _output_dir_path;
};

}  // namespace idrc
#endif  // IDR_SRC_CFG_CONFIG_H_
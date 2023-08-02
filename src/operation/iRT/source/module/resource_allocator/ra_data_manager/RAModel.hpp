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
#pragma once

#include "RAGCell.hpp"
#include "RAModelStat.hpp"
#include "RANet.hpp"

namespace irt {

class RAModel
{
 public:
  RAModel() = default;
  ~RAModel() = default;
  // getter
  std::vector<RANet>& get_ra_net_list() { return _ra_net_list; }
  std::vector<RAGCell>& get_ra_gcell_list() { return _ra_gcell_list; }
  std::vector<double>& get_result_list() { return _result_list; }
  std::vector<double>& get_nabla_f_col() { return _nabla_f_col; }
  std::vector<double>& get_nabla_f_row() { return _nabla_f_row; }
  double get_alpha() const { return _alpha; }
  RAModelStat& get_ra_model_stat() { return _ra_model_stat; }
  irt_int get_curr_outer_iter() { return _curr_outer_iter; }
  irt_int get_curr_inner_iter() { return _curr_inner_iter; }
  // setter
  void set_ra_net_list(const std::vector<RANet>& ra_net_list) { _ra_net_list = ra_net_list; }
  void set_ra_gcell_list(const std::vector<RAGCell>& ra_gcell_list) { _ra_gcell_list = ra_gcell_list; }
  void set_result_list(const std::vector<double>& result_list) { _result_list = result_list; }
  void set_nabla_f_col(const std::vector<double>& nabla_f_col) { _nabla_f_col = nabla_f_col; }
  void set_nabla_f_row(const std::vector<double>& nabla_f_row) { _nabla_f_row = nabla_f_row; }
  void set_alpha(const double alpha) { _alpha = alpha; }
  void set_ra_model_stat(const RAModelStat& ra_model_stat) { _ra_model_stat = ra_model_stat; }
  void set_curr_outer_iter(const irt_int curr_outer_iter) { _curr_outer_iter = curr_outer_iter; }
  void set_curr_inner_iter(const irt_int curr_inner_iter) { _curr_inner_iter = curr_inner_iter; }
  // function

 private:
  std::vector<RANet> _ra_net_list;
  std::vector<RAGCell> _ra_gcell_list;
  // run time object
  std::vector<double> _result_list;
  std::vector<double> _nabla_f_col;
  std::vector<double> _nabla_f_row;
  double _alpha = 0;
  RAModelStat _ra_model_stat;
  irt_int _curr_outer_iter = -1;
  irt_int _curr_inner_iter = -1;
};
}  // namespace irt

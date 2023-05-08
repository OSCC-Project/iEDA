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
  // setter
  void set_ra_net_list(const std::vector<RANet>& ra_net_list) { _ra_net_list = ra_net_list; }
  void set_alpha(const double alpha) { _alpha = alpha; }

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
};
}  // namespace irt

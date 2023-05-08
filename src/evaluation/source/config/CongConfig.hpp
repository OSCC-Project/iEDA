#ifndef SRC_EVALUATOR_SOURCE_WRAPPER_CONFIG_CONGCONFIG_HPP_
#define SRC_EVALUATOR_SOURCE_WRAPPER_CONFIG_CONGCONFIG_HPP_

#include <string>

namespace eval {

class CongConfig
{
 public:
  CongConfig() = default;
  ~CongConfig() = default;

  // getter
  bool enable_eval() const { return _enable_eval; }
  std::string get_eval_type() const { return _eval_type; }
  int32_t get_bin_cnt_x() const { return _bin_cnt_x; }
  int32_t get_bin_cnt_y() const { return _bin_cnt_y; }
  int32_t get_tile_size_x() const { return _tile_size_x; }
  int32_t get_tile_size_y() const { return _tile_size_y; }
  std::string get_output_dir() const { return _output_dir; }
  std::string get_output_filename() const { return _output_filename; }

  // setter
  void enable_eval(const bool& enable_eval) { _enable_eval = enable_eval; }
  void set_eval_type(const std::string& eval_type) { _eval_type = eval_type; }
  void set_bin_cnt_x(const int32_t& bin_cnt_x) { _bin_cnt_x = bin_cnt_x; }
  void set_bin_cnt_y(const int32_t& bin_cnt_y) { _bin_cnt_y = bin_cnt_y; }
  void set_tile_size_x(const int32_t& tile_size_x) { _tile_size_x = tile_size_x; }
  void set_tile_size_y(const int32_t& tile_size_y) { _tile_size_y = tile_size_y; }
  void set_output_dir(const std::string& output_dir) { _output_dir = output_dir; }
  void set_output_filename(const std::string& output_filename) { _output_filename = output_filename; }

 private:
  bool _enable_eval;
  std::string _eval_type;
  int32_t _bin_cnt_x;
  int32_t _bin_cnt_y;
  int32_t _tile_size_x;
  int32_t _tile_size_y;
  std::string _output_dir;
  std::string _output_filename;
};
}  // namespace eval

#endif  // SRC_EVALUATOR_SOURCE_WRAPPER_CONFIG_CONGCONFIG_HPP_

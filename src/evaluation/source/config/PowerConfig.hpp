#ifndef SRC_EVALUATOR_SOURCE_WRAPPER_CONFIG_POWERCONFIG_HPP_
#define SRC_EVALUATOR_SOURCE_WRAPPER_CONFIG_POWERCONFIG_HPP_

#include <string>

namespace eval {

class PowerConfig
{
 public:
  PowerConfig() = default;
  ~PowerConfig() = default;

  // getter
  bool enable_eval() const { return _enable_eval; }
  std::string get_output_dir() const { return _output_dir; }
  std::string get_output_filename() const { return _output_filename; }

  // setter
  void enable_eval(const bool& enable_eval) { _enable_eval = enable_eval; }
  void set_output_dir(const std::string& output_dir) { _output_dir = output_dir; }
  void set_output_filename(const std::string& output_filename) { _output_filename = output_filename; }

 private:
  bool _enable_eval;
  std::string _output_dir;
  std::string _output_filename;
};
}  // namespace eval

#endif  // SRC_EVALUATOR_SOURCE_WRAPPER_CONFIG_POWERCONFIG_HPP_

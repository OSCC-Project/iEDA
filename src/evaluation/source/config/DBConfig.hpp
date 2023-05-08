#ifndef SRC_EVALUATOR_SOURCE_WRAPPER_CONFIG_DBCONFIG_HPP_
#define SRC_EVALUATOR_SOURCE_WRAPPER_CONFIG_DBCONFIG_HPP_

#include <string>
#include <vector>

namespace eval {

class DBConfig
{
 public:
  DBConfig() = default;
  ~DBConfig() = default;

  // getter.
  bool enable_wrapper() const { return _enable_wrapper; }
  std::string get_source() const { return _source; }
  std::string get_separator() const { return _separator; }
  std::vector<std::string> get_lef_file_list() const { return _lef_file_list; }
  std::string get_def_file() const { return _def_file; }

  // setter.
  void enable_wrapper(const bool& enable_wrapper) { _enable_wrapper = enable_wrapper; }
  void set_source(const std::string& source) { _source = source; }
  void set_separator(const std::string& separator) { _separator = separator; }
  void set_lef_file_list(const std::vector<std::string>& lef_file_list) { _lef_file_list = lef_file_list; }
  void set_def_file(const std::string& def_file) { _def_file = def_file; }

 private:
  bool _enable_wrapper;
  std::string _source;
  std::string _separator;
  std::vector<std::string> _lef_file_list;
  std::string _def_file;
};

}  // namespace eval

#endif  // SRC_EVALUATOR_SOURCE_WRAPPER_CONFIG_DBCONFIG_HPP_

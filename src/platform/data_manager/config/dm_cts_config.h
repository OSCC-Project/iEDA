#pragma once
/**
 * @File Name: dm_cts_config.h
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2022-04-15
 *
 */
#include <map>
#include <string>
#include <vector>

#include "json.hpp"
using json = nlohmann::json;

using namespace std;
namespace idm {

class CtsConfig
{
 public:
  CtsConfig(string config_path) { initConfig(config_path); }
  ~CtsConfig() = default;

  /// getter
  const string get_read_cts_data() const { return _read_cts_data; }
  const string get_write_cts_data() const { return _write_cts_data; }
  const string get_cts_data_path() const { return _cts_data_path; }
  bool is_read_cts_data() { return _read_cts_data == "ON" ? true : false; }
  bool is_write_cts_data() { return _write_cts_data == "ON" ? true : false; }

  /// setter
  void set_read_cts_data(string read_cts_data) { _read_cts_data = read_cts_data; }
  void set_write_cts_data(string write_cts_data) { _write_cts_data = write_cts_data; }
  void set_cts_data_path(const string cts_data_path) { _cts_data_path = cts_data_path; }

  /// function
  bool initConfig(string config_path);

  /// check file exist
  bool checkFilePath(string path);

 private:
  string _config_path;

  string _read_cts_data;
  string _write_cts_data;
  string _cts_data_path;
};

}  // namespace idm

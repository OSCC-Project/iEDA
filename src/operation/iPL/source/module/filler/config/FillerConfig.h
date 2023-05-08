/*
 * @Author: WenruiWu
 * @Date: 2022-04-15
 * @LastEditTime: 2022-04-15
 * @LastEditors: WenruiWu
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/operator/detail_placer/config/FillerConfig.hh
 * Contact :
 */

#ifndef IPL_OPERATOR_FILLER_CONFIG_H
#define IPL_OPERATOR_FILLER_CONFIG_H

#include <string>
#include <vector>

namespace ipl {

class FillerConfig
{
 public:
  FillerConfig() = default;
  FillerConfig(const FillerConfig& other) = default;
  FillerConfig(FillerConfig&& other) = default;
  ~FillerConfig() = default;

  FillerConfig& operator=(const FillerConfig& other) = default;
  FillerConfig& operator=(FillerConfig&& other) = default;

  // getter.
  int32_t get_thread_num() const { return _thread_num;}
  std::vector<std::vector<std::string>> get_filler_group_list() { return _filler_group_list; }
  int32_t get_min_filler_width() { return _min_filler_width; }
  
  // setter.
  void set_thread_num(int32_t num_thread) { _thread_num = num_thread;}
  void set_filler_group_list(std::vector<std::vector<std::string>> filler_group_list) { _filler_group_list = filler_group_list; }
  void add_filler_name_list(std::vector<std::string> fill_name_list) { _filler_group_list.push_back(fill_name_list); }
  void set_min_filler_width(int32_t min_filler_width) { _min_filler_width = min_filler_width; }

 private:
  int32_t _thread_num;

  // filler cell name list
  std::vector<std::vector<std::string>> _filler_group_list;
  int32_t _min_filler_width;
};

}  // namespace ipl

#endif
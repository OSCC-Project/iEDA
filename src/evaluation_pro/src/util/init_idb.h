/*
 * @FilePath: init_idb.h
 * @Author: Yihang Qiu (qiuyihang23@mails.ucas.ac.cn)
 * @Date: 2024-09-02 14:14:04
 * @Description:
 */
#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace ieval {

class InitIDB
{
 public:
  InitIDB();
  ~InitIDB();

  static InitIDB* getInst();
  static void destroyInst();

  void initPointSets();
  std::vector<std::vector<std::pair<int32_t, int32_t>>> getPointSets() { return _point_sets; }
  int32_t getDesignUnit();
  std::map<std::string, std::vector<std::pair<int32_t, int32_t>>> getNamePointSet() { return _name_point_set; }

 private:
  static InitIDB* _init_idb;

  std::vector<std::vector<std::pair<int32_t, int32_t>>> _point_sets;
  std::map<std::string, std::vector<std::pair<int32_t, int32_t>>> _name_point_set;
};
}  // namespace ieval

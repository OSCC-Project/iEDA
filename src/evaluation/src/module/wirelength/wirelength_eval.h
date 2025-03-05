/*
 * @FilePath: wirelength_eval.h
 * @Author: Yihang Qiu (qiuyihang23@mails.ucas.ac.cn)
 * @Date: 2024-08-28 19:51:53
 * @Description:
 */

#pragma once

#include <map>

#include "wirelength_db.h"

namespace ieval {

class WirelengthEval
{
 public:
  WirelengthEval();
  ~WirelengthEval();
  static WirelengthEval* getInst();
  static void destroyInst();

  int32_t evalTotalHPWL(PointSets point_sets);
  int32_t evalTotalFLUTE(PointSets point_sets);
  int32_t evalTotalHTree(PointSets point_sets);
  int32_t evalTotalVTree(PointSets point_sets);

  int32_t evalTotalHPWL();
  int32_t evalTotalFLUTE();
  int32_t evalTotalHTree();
  int32_t evalTotalVTree();
  int32_t evalTotalEGRWL();

  int32_t evalNetHPWL(PointSet point_set);
  int32_t evalNetFLUTE(PointSet point_set);
  int32_t evalNetHTree(PointSet point_set);
  int32_t evalNetVTree(PointSet point_set);

  int32_t evalPathHPWL(PointSet point_set, PointPair point_pair);
  int32_t evalPathFLUTE(PointSet runEGRpoint_set, PointPair point_pair);
  int32_t evalPathHTree(PointSet point_set, PointPair point_pair);
  int32_t evalPathVTree(PointSet point_set, PointPair point_pair);

  float evalTotalEGRWL(std::string guide_path);
  float evalNetEGRWL(std::string guide_path, std::string net_name);
  float evalPathEGRWL(std::string guide_path, std::string net_name, std::string load_name);

  int32_t getDesignUnit();
  std::map<std::string, std::vector<std::pair<int32_t, int32_t>>> getNamePointSet();
  std::vector<std::pair<int32_t, int32_t>> getNetPointSet(std::string net_name);

  void initIDB();
  void destroyIDB();
  void initEGR();
  void destroyEGR();
  void initFlute();
  void destroyFlute();

  void evalNetInfo();
  void evalNetFlute();
  int32_t findHPWL(std::string net_name);
  int32_t findFLUTE(std::string net_name);
  int32_t findGRWL(std::string net_name);

  std::string getEGRDirPath();

 private:
  static WirelengthEval* _wirelength_eval;

  std::map<std::string, int32_t> _name_hpwl;
  std::map<std::string, int32_t> _name_flute;
  std::map<std::string, int32_t> _name_grwl;
};

}  // namespace ieval
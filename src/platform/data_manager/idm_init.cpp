/**
 * @File Name: dm_init.cpp
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2022-04-15
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "idm.h"

namespace idm {
bool DataManager::initLef(vector<string> lef_paths)
{
  _idb_lef_service = _idb_builder->buildLef(lef_paths);
  _layout = get_idb_layout();

  return _idb_lef_service == nullptr ? false : true;
}

bool DataManager::initDef(string def_path)
{
  _idb_def_service = _idb_builder->buildDef(def_path);
  _design = get_idb_design();

  /// make original coordinate on (0,0)
  if (isNeedTransformByDie()) {
    /// transform
    transformByDie();
  }

  return _idb_def_service == nullptr ? false : true;
}

bool DataManager::initVerilog(string verilog_path, string top_module)
{
  _idb_def_service = _idb_builder->buildVerilog(verilog_path, top_module);
  _design = get_idb_design();

  return _idb_def_service == nullptr ? false : true;
}

}  // namespace idm

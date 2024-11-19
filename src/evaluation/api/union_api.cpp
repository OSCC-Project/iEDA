/*
 * @FilePath: union_api.cpp
 * @Author: Yihang Qiu (qiuyihang23@mails.ucas.ac.cn)
 * @Date: 2024-10-11 15:37:27
 * @Description:
 */

#include "union_api.h"

#include "init_egr.h"
#include "init_flute.h"
#include "init_idb.h"

namespace ieval {

#define UNION_EVAL_INIT_IDB_INST (ieval::InitIDB::getInst())
#define UNION_EVAL_INIT_FLUTE_INST (ieval::InitFlute::getInst())
#define UNION_EVAL_INIT_EGR_INST (ieval::InitEGR::getInst())

UnionAPI* UnionAPI::_union_api_inst = nullptr;

UnionAPI::UnionAPI()
{
}

UnionAPI::~UnionAPI()
{
}

UnionAPI* UnionAPI::getInst()
{
  if (_union_api_inst == nullptr) {
    _union_api_inst = new UnionAPI();
  }

  return _union_api_inst;
}

void UnionAPI::destroyInst()
{
  if (_union_api_inst != nullptr) {
    delete _union_api_inst;
    _union_api_inst = nullptr;
  }
}

void UnionAPI::initIDB()
{
  UNION_EVAL_INIT_IDB_INST->initPointSets();
  UNION_EVAL_INIT_IDB_INST->initDensityDB();
  UNION_EVAL_INIT_IDB_INST->initCongestionDB();
}

void UnionAPI::initEGR(bool enable_timing)
{
  UNION_EVAL_INIT_EGR_INST->runEGR(enable_timing);
}

void UnionAPI::initFlute()
{
  UNION_EVAL_INIT_FLUTE_INST->readLUT();
}

void UnionAPI::destroyIDB()
{
  UNION_EVAL_INIT_IDB_INST->destroyInst();
}

void UnionAPI::destroyEGR()
{
  UNION_EVAL_INIT_EGR_INST->destroyInst();
}

void UnionAPI::destroyFlute()
{
  UNION_EVAL_INIT_FLUTE_INST->destroyInst();
}

}  // namespace ieval

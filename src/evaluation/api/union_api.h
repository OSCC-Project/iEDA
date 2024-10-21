/*
 * @FilePath: union_api.h
 * @Author: Yihang Qiu (qiuyihang23@mails.ucas.ac.cn)
 * @Date: 2024-10-11 15:37:27
 * @Description:
 */

#pragma once

namespace ieval {

#define UNION_API_INST (ieval::UnionAPI::getInst())

class UnionAPI
{
 public:
  UnionAPI();
  ~UnionAPI();
  static UnionAPI* getInst();
  static void destroyInst();

  void initIDB();
  void initEGR(bool enable_timing = false);
  void initFlute();

  void destroyIDB();
  void destroyEGR();
  void destroyFlute();

 private:
  static UnionAPI* _union_api_inst;
};

}  // namespace ieval
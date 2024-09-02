/*
 * @FilePath: init_flute.h
 * @Author: Yihang Qiu (qiuyihang23@mails.ucas.ac.cn)
 * @Date: 2024-08-24 15:37:27
 * @Description:
 */

#pragma once

#include "flute.h"

namespace ieval {

class InitFlute
{
 public:
  InitFlute();
  ~InitFlute();
  static InitFlute* getInst();
  static void destroyInst();

  void readLUT();
  void deleteLUT();
  void printTree(Flute::Tree flutetree);
  void freeTree(Flute::Tree flutetree);
  Flute::Tree flute(int d, int* x, int* y, int acc);

 private:
  static InitFlute* _init_flute;
};

}  // namespace ieval
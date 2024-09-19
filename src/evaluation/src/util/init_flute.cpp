/*
 * @FilePath: init_flute.cpp
 * @Author: Yihang Qiu (qiuyihang23@mails.ucas.ac.cn)
 * @Date: 2024-08-24 15:37:27
 * @Description:
 */

#include "init_flute.h"

namespace ieval {

InitFlute* InitFlute::_init_flute = nullptr;

InitFlute::InitFlute()
{
}

InitFlute::~InitFlute()
{
}

InitFlute* InitFlute::getInst()
{
  if (_init_flute == nullptr) {
    _init_flute = new InitFlute();
  }
  return _init_flute;
}

void InitFlute::destroyInst()
{
  if (_init_flute != nullptr) {
    delete _init_flute;
    _init_flute = nullptr;
  }
}

void InitFlute::readLUT()
{
  Flute::readLUT();
}

void InitFlute::deleteLUT()
{
  Flute::deleteLUT();
}

void InitFlute::printTree(Flute::Tree flutetree)
{
  Flute::printtree(flutetree);
}

void InitFlute::freeTree(Flute::Tree flutetree)
{
  Flute::free_tree(flutetree);
}

Flute::Tree InitFlute::flute(int d, int* x, int* y, int acc)
{
  return Flute::flute(d, x, y, acc);
}

}  // namespace ieval
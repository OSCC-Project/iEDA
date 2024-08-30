/*
 * @FilePath: init_flute.cpp
 * @Author: Yihang Qiu (qiuyihang23@mails.ucas.ac.cn)
 * @Date: 2024-08-24 15:37:27
 * @Description: 
 */


#include "init_flute.h"

namespace ieval {

InitFlute::InitFlute()
{
}

InitFlute::~InitFlute()
{
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
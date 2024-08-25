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
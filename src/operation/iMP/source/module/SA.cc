#include "SA.hh"

#include <cstring>
#include <map>
#include <vector>

namespace imp {

bool pack_sp(int size, int* pos, int* neg, int* w, int* h, int* lx, int* ly, int& W, int& H)
{
  std::map<int, int> BST;

  auto find = [&](int index) {
    int loc;
    auto iter = BST.lower_bound(index);
    if (iter != BST.begin()) {
      iter--;
      loc = (*iter).second;
    } else
      loc = 0;
    return loc;
  };

  auto remove = [&](int index, int length) {
    auto endIter = BST.end();
    auto iter = BST.find(index);
    auto nextIter = iter;
    ++nextIter;
    if (nextIter != BST.end()) {
      ++iter;
      while (true) {
        ++nextIter;
        if ((*iter).second < length)
          BST.erase(iter);
        if (nextIter == endIter)
          break;
        iter = nextIter;
      }
    }
  };

  std::vector<int> match(size);

  auto pack = [&](int size, int* pos, int* neg, int* loc, int* weight) {
    BST.clear();
    BST[0] = 0;
    for (int i{0}; i < size; i++) {
      match[neg[i]] = i;
    }

    float t{0.f};
    for (int i{0}; i < size; ++i) {
      int p = match[pos[i]];
      loc[pos[i]] = find(p);
      t = loc[pos[i]] + weight[pos[i]];
      BST[p] = t;
      remove(p, t);
    }
    return find(size);
  };

  W = pack(size, pos, neg, lx, w);

  int rpos_sp[size];

  int iter = 0;
  int riter = size;
  while (iter < size) {
    rpos_sp[--riter] = pos[iter++];
  }
  H = pack(size, rpos_sp, neg, ly, h);

  return true;
}

}  // namespace imp

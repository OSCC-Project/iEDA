#include "SA.hh"

#include <cstring>
#include <map>
#include <vector>

#include "Evaluator.hh"
namespace imp {
template <typename T>
SequencePair<T>::SequencePair(int s, int* ps, int* ns, const T& t) : size(s), pos_seq(new int[size]), neg_seq(new int[size]), Tp(t)
{
  std::memcpy(pos_seq, ps, size * sizeof(int));
  std::memcpy(neg_seq, ns, size * sizeof(int));
}
template <typename T>
SequencePair<T>::SequencePair(const SequencePair<T>& other) : size(other.size), pos_seq(new int[size]), neg_seq(new int[size]), Tp(other.Tp)
{
  std::memcpy(pos_seq, other.pos_seq, size * sizeof(int));
  std::memcpy(neg_seq, other.neg_seq, size * sizeof(int));
}
template <typename T>
SequencePair<T>& SequencePair<T>::operator=(const SequencePair<T>& other)
{
  if (this != &other) {
    if (size != other.size) {
      size = other.size;
      delete[] pos_seq;
      delete[] neg_seq;
      pos_seq = new int[size];
      neg_seq = new int[size];
    }
    std::memcpy(pos_seq, other.pos_seq, size * sizeof(int));
    std::memcpy(neg_seq, other.neg_seq, size * sizeof(int));
    Tp = other.Tp;
  }
  return *this;
}
template <typename T>
SequencePair<T>::~SequencePair()
{
  delete[] pos_seq;
  delete[] neg_seq;
}
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

double evaluate(SequencePair<NetList> sp)
{
  const NetList& netlist = sp.Tp;
  int W, H;
  // pack_sp(netlist.num_v, sp.pos_seq, sp.neg_seq, netlist.w, netlist.h, netlist.x, netlist.y, W, H);

  return 0.0;
}

}  // namespace imp

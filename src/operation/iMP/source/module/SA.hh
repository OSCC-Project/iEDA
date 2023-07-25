/**
 * @file SA.hh
 * @author Fuxing Huang (fxxhuang@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-07-12
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef IMP_SA_H
#define IMP_SA_H
// #include "Annealer.hh"
#include "NetList.hh"
namespace imp {

template <typename T>
struct SequencePair
{
  SequencePair(int s, int* ps, int* ns, const T&);
  SequencePair(const SequencePair<T>& other);
  SequencePair& operator=(const SequencePair<T>& other);
  ~SequencePair();
  int size;
  int* pos_seq;
  int* neg_seq;
  const T& Tp;
};

template <typename T>
bool pack_sp(int size, int* pos, int* neg, int* w, int* h, int* lx, int* ly, int& W, int& H);
double evaluate(SequencePair<NetList> sp);

}  // namespace imp

#endif
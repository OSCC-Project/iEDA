/**
 * @file SeqPair.hh
 * @author Fuxing Huang (fxxhuang@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-08-07
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef IMP_SEQPAIR_H
#define IMP_SEQPAIR_H
#include <cstdint>
#include <functional>
#include <map>
#include <random>
#include <span>
#include <tuple>
#include <vector>

#include "Evaluator.hh"
namespace imp {

class NetList;

struct SeqPair
{
  SeqPair(const size_t sz);
  SeqPair(const SeqPair& other);
  ~SeqPair() = default;
  SeqPair& operator=(const SeqPair& other);
  size_t size{0};
  std::vector<size_t> pos{};
  std::vector<size_t> neg{};
};
auto makeRandomSeqPair(size_t sz) -> SeqPair
{
  SeqPair sp(sz);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::shuffle(std::begin(sp.pos), std::end(sp.pos), gen);
  std::shuffle(std::begin(sp.neg), std::end(sp.neg), gen);
  return sp;
}

template <typename T>
struct SpLocation
{
  std::pair<T, T> operator()(const SeqPair&, const std::vector<T>& width, const std::vector<T>& height, std::vector<T>& x,
                             std::vector<T>& y, bool is_left = true, bool is_bottom = true);

 private:
  T find(size_t id);
  void remove(size_t id, T);
  T pack(const std::vector<size_t>& pos, const std::vector<size_t>& neg, const std::vector<T>& weight, std::vector<T>& loc);
  std::map<size_t, T> _bst;
  std::vector<size_t> _reverse_pos{};
  std::vector<size_t> _reverse_neg{};
  std::vector<size_t> _match{};
};

struct SpEvaluate
{
  double operator()(const SeqPair&);

  SpEvaluate(const NetList& netlist) : _netlist(netlist) {}

 private:
  SpLocation<int64_t> _get_locaton{};
  std::vector<double> _pin_x_buffer{};
  std::vector<double> _pin_y_buffer{};
  std::vector<int64_t> _lx_buffer{};
  std::vector<int64_t> _ly_buffer{};
  const NetList& _netlist{};
};

auto makeSeqPairEvalFn(const NetList& netlist) -> std::function<double(const SeqPair&)>
{
  return SpEvaluate(netlist);
}

struct SpAction
{
  void operator()(SeqPair& sp);
  SpAction(size_t size);

 private:
  std::mt19937 _gen;
  std::uniform_int_distribution<size_t> _random_index;
  std::discrete_distribution<size_t> _random_probaility;
  std::vector<std::function<void(SeqPair&)>> _actions;
};

}  // namespace imp
#include "SeqPair.tpp"
#endif
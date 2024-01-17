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

  SpEvaluate(const NetList& netlist) : _netlist(netlist)
  {
    _pin_x_buffer.resize(_netlist.pin_x_off.size());
    _pin_y_buffer.resize(_netlist.pin_x_off.size());
    _lx_buffer = _netlist.lx;
    _ly_buffer = _netlist.ly;
    for (size_t i = 0; i < _netlist.pin2vertex.size(); i++) {
      size_t j = _netlist.pin2vertex[i];
      if (j >= _netlist.num_moveable) {
        _pin_x_buffer[i] = _lx_buffer[j];
        _pin_y_buffer[i] = _ly_buffer[j];
      }
    }
    SeqPair sp(_netlist.num_moveable);
    std::random_device rd;
    std::mt19937 gen(rd());
    double sum_wl = 0;
    for (size_t i = 0; i < 100; i++) {
      std::shuffle(std::begin(sp.pos), std::end(sp.pos), gen);
      std::shuffle(std::begin(sp.neg), std::end(sp.neg), gen);
      packing(sp);
      sum_wl += _wl;
    }
    _avg_wl = sum_wl / 100;
  }
  void packing(const SeqPair&);

  const std::vector<double>& get_pin_x() { return _pin_x_buffer; }
  const std::vector<double>& get_pin_y() { return _pin_y_buffer; }
  const std::vector<int64_t>& get_lx() { return _lx_buffer; }
  const std::vector<int64_t>& get_ly() { return _ly_buffer; }
  double get_wl() { return _wl; }
  double get_bound_dx() { return _bound_dx; }
  double get_bound_dy() { return _bound_dy; }

 private:
  SpLocation<int64_t> _get_locaton{};
  std::vector<double> _pin_x_buffer{};
  std::vector<double> _pin_y_buffer{};
  std::vector<int64_t> _lx_buffer{};
  std::vector<int64_t> _ly_buffer{};
  double _wl{};
  double _avg_wl{};
  int64_t _bound_dx{};
  int64_t _bound_dy{};
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
  std::pair<size_t, size_t> index_pair()
  {
    size_t first = _random_index(_gen);
    size_t second = _random_index(_gen);
    while (first == second) {
      second = _random_index(_gen);
    }
    return std::make_pair(first, second);
  }
  static void _pos_swap(SeqPair& sp, SpAction* action);
  static void _neg_swap(SeqPair& sp, SpAction* action);
  static void _double_swap(SeqPair& sp, SpAction* action);
  static void _pos_insert(SeqPair& sp, SpAction* action);
  static void _neg_insert(SeqPair& sp, SpAction* action);

  std::mt19937 _gen;
  std::uniform_int_distribution<size_t> _random_index;
  std::discrete_distribution<size_t> _random_probaility;
  std::vector<std::function<void(SeqPair&, SpAction*)>> _actions;
};

struct SpPlot
{
  bool operator()(const SeqPair& sp, std::string filename);
  SpPlot(const NetList& netlist, int64_t xlim, int64_t ylim) : _netlist(netlist), _xlim(xlim), _ylim(ylim) {}

 private:
  const NetList& _netlist{};
  int64_t _xlim;
  int64_t _ylim;
};

}  // namespace imp
#include "SeqPair.tpp"
#endif
#include <algorithm>
#include <ranges>

#include "Evaluator.hh"
#include "NetList.hh"
#include "SeqPair.hh"
namespace imp {
SeqPair::SeqPair(const size_t sz) : size(sz)  //, max_status(max_st)
{
  pos.resize(sz);
  neg.resize(sz);
  // status.resize(sz);
  std::iota(pos.begin(), pos.end(), 0);
  std::iota(neg.begin(), neg.end(), 0);
  // std::uniform_int_distribution<uint8_t> st(0, max_status - 1);
  // for (auto& i : status) {
  //   i = st(gen);
  // }
}

SeqPair::SeqPair(const SeqPair& other)
{
  size = other.size;
  // max_status = other.max_status;
  pos = other.pos;
  neg = other.neg;
  // status = other.status;
}

SeqPair& SeqPair::operator=(const SeqPair& other)
{
  if (size == other.size) {
    std::copy(other.pos.begin(), other.pos.end(), pos.begin());
    std::copy(other.neg.begin(), other.neg.end(), neg.begin());
    // std::copy(other.status.begin(), other.status.end(), status.begin());
  } else {
    pos = other.pos;
    neg = other.neg;
    // status = other.status;
    size = other.size;
  }
  // max_status = other.max_status;
  return *this;
}

template <typename T>
std::pair<T, T> SpLocation<T>::operator()(const SeqPair& sp, const std::vector<T>& width, const std::vector<T>& height, std::vector<T>& x,
                                          std::vector<T>& y, bool is_left, bool is_bottom)
{
  T bound_x{0};
  T bound_y{0};
  if (sp.size > _match.size()) {
    _match.resize(sp.size);
    _reverse_pos.resize(sp.size);
    _reverse_neg.resize(sp.size);
  }
  std::fill(_match.begin(), _match.end(), 0);
  if (is_left) {
    bound_x = pack(sp.pos, sp.neg, width, x);
  } else {
    std::reverse_copy(std::begin(sp.pos), std::end(sp.pos), std::begin(_reverse_pos));
    std::reverse_copy(std::begin(sp.neg), std::end(sp.neg), std::begin(_reverse_neg));
    bound_x = pack(_reverse_pos, _reverse_neg, width, x);
    for (size_t i = 0; i < sp.size; i++) {
      x[i] = bound_x - x[i] - width[i];
    }
  }
  std::fill(_match.begin(), _match.end(), 0);
  if (is_bottom) {
    if (is_left)
      std::reverse_copy(std::begin(sp.pos), std::end(sp.pos), std::begin(_reverse_pos));
    bound_y = pack(_reverse_pos, sp.neg, height, y);
  } else {
    if (is_left)
      std::reverse_copy(std::begin(sp.neg), std::end(sp.neg), std::begin(_reverse_neg));
    bound_y = pack(sp.pos, _reverse_neg, height, y);
    for (size_t i = 0; i < sp.size; i++) {
      y[i] = bound_y - y[i] - height[i];
    }
  }
  return std::make_pair(bound_x, bound_y);
}

template <typename T>
inline T SpLocation<T>::find(size_t id)
{
  T loc{0};
  auto iter = _bst.lower_bound(id);
  if (iter != _bst.begin()) {
    iter--;
    loc = (*iter).second;
  } else
    loc = 0;
  return loc;
}

template <typename T>
inline void SpLocation<T>::remove(size_t id, T length)
{
  auto endIter = _bst.end();
  auto iter = _bst.find(id);
  auto nextIter = iter;
  ++nextIter;
  if (nextIter != _bst.end()) {
    ++iter;
    while (true) {
      ++nextIter;
      if ((*iter).second < length)
        _bst.erase(iter);
      if (nextIter == endIter)
        break;
      iter = nextIter;
    }
  }
}

template <typename T>
inline T SpLocation<T>::pack(const std::vector<size_t>& pos, const std::vector<size_t>& neg, const std::vector<T>& weight,
                             std::vector<T>& loc)
{
  _bst.clear();
  _bst[0] = 0;
  size_t size = pos.size();
  for (size_t i = 0; i < size; ++i) {
    _match[neg[i]] = i;
  }
  T t{0};
  for (size_t i = 0; i < size; ++i) {
    size_t p = _match[pos[i]];
    loc[pos[i]] = find(p);
    t = loc[pos[i]] + weight[pos[i]];
    _bst[p] = t;
    remove(p, t);
  }
  T length = find(size);
  return length;
}

double SpEvaluate::operator()(const SeqPair& sp)
{
  if (_pin_x_buffer.empty()) {
    _pin_x_buffer.resize(_netlist._pin_x_off.size());
    _pin_y_buffer.resize(_netlist._pin_x_off.size());
    _lx_buffer = _netlist._lx;
    _ly_buffer = _netlist._ly;
  }
  auto [bound_w, bound_h] = _get_locaton(sp, _netlist._dx, _netlist._dy, _lx_buffer, _ly_buffer);
  for (size_t i = 0; i < _netlist._pin2vertex.size(); i++) {
    size_t j = _netlist._pin2vertex[i];
    _pin_x_buffer[i] = static_cast<double>(_netlist._region_lx + _lx_buffer[j] + _netlist._pin_x_off[i]) + _netlist._dx[j] / 2.0;
    _pin_y_buffer[i] = static_cast<double>(_netlist._region_ly + _ly_buffer[j] + _netlist._pin_y_off[i]) + _netlist._dy[j] / 2.0;
  }
  double wl = hpwl(_pin_x_buffer, _pin_y_buffer, _netlist._net_span);
  double A = std::max(bound_w - _netlist._region_dx, int64_t(0)) + std::max(bound_h - _netlist._region_dy, int64_t(0));

  return A;
}

SpAction::SpAction(size_t size)
{
  std::random_device r;
  _gen = std::mt19937(r());
  _random_index = std::uniform_int_distribution<size_t>(size_t(0), size_t(size - 1));

  auto index_pair = [&]() {
    size_t first = _random_index(_gen);
    size_t second = _random_index(_gen);
    while (first == second) {
      second = _random_index(_gen);
    }
    return std::make_pair(first, second);
  };

  auto pos_swap = [&](SeqPair& sp) {
    auto [first, second] = index_pair();
    std::swap(sp.pos[first], sp.pos[second]);
  };

  auto neg_swap = [&](SeqPair& sp) {
    auto [first, second] = index_pair();
    std::swap(sp.neg[first], sp.neg[second]);
  };

  auto double_swap = [&](SeqPair& sp) {
    auto [first1, second1] = index_pair();
    std::swap(sp.pos[first1], sp.pos[second1]);
    auto [first2, second2] = index_pair();
    std::swap(sp.neg[first2], sp.neg[second2]);
  };

  auto pos_insert = [&](SeqPair& sp) {
    auto [first, second] = index_pair();
    size_t val = sp.pos[first];
    sp.pos.erase(std::begin(sp.pos) + first);
    sp.pos.insert(std::begin(sp.pos) + second, val);
  };

  auto neg_insert = [&](SeqPair& sp) {
    auto [first, second] = index_pair();
    size_t val = sp.neg[first];
    sp.neg.erase(std::begin(sp.neg) + first);
    sp.neg.insert(std::begin(sp.neg) + second, val);
  };

  _actions = {pos_swap, neg_swap, double_swap, pos_insert, neg_insert};

  _random_probaility = std::discrete_distribution<size_t>({150, 150, 80, 100, 100});
}

void SpAction::operator()(SeqPair& sp)
{
  _actions[_random_probaility(_gen)](sp);
}

}  // namespace imp

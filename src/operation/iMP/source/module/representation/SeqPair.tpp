#include <algorithm>
#include <ranges>

#include "Evaluator.hh"
#include "PyPlot.hh"
#include "SeqPair.hh"
namespace imp {
SeqPair::SeqPair(const size_t sz) : size(sz)  //, max_status(max_st)
{
  pos.resize(sz);
  neg.resize(sz);
  std::iota(pos.begin(), pos.end(), 0);
  std::iota(neg.begin(), neg.end(), 0);

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
  packing(sp);

  double norm_wl = _wl / _avg_wl;

  double norm_bound = std::max((_bound_dx * 1. - _netlist.region_dx * 1.) / (_netlist.region_dx * 1.), 0.)
                      + std::max((_bound_dy * 1. - _netlist.region_dy * 1.) / (_netlist.region_dy * 1.), 0.);
  // double norm_area = _bound_dx * _bound_dy / _netlist.sum_vertex_area;
  // double norm_area = 0.;
  double weight_hpwl = 0.7;

  return (1 - weight_hpwl) * norm_bound + weight_hpwl * norm_wl;
}
void SpEvaluate::packing(const SeqPair& sp)
{
  auto [bound_w, bound_h] = _get_locaton(sp, _netlist.dx, _netlist.dy, _lx_buffer, _ly_buffer);
  _bound_dx = bound_w;
  _bound_dy = bound_h;
  for (size_t i = 0; i < _netlist.pin2vertex.size(); i++) {
    size_t j = _netlist.pin2vertex[i];
    if (_netlist.pin2vertex[i] >= sp.size)
      continue;
    _pin_x_buffer[i] = static_cast<double>(_netlist.region_lx + _lx_buffer[j] + _netlist.pin_x_off[i]) + _netlist.dx[j] / 2.0;
    _pin_y_buffer[i] = static_cast<double>(_netlist.region_ly + _ly_buffer[j] + _netlist.pin_y_off[i]) + _netlist.dy[j] / 2.0;
  }
  _wl = hpwl(_pin_x_buffer, _pin_y_buffer, _netlist.net_span, 16);
}

SpAction::SpAction(size_t size)
{
  std::random_device r;
  _gen = std::mt19937(r());
  _random_index = std::uniform_int_distribution<size_t>(size_t(0), size_t(size - 1));
  _actions = {_pos_swap, _neg_swap, _double_swap, _pos_insert, _neg_insert};
  _random_probaility = std::discrete_distribution<size_t>({150, 150, 80, 100, 100});
}

void SpAction::operator()(SeqPair& sp)
{
  _actions[_random_probaility(_gen)](sp, this);
}

void SpAction::_pos_swap(SeqPair& sp, SpAction* action)
{
  auto [first, second] = action->index_pair();
  std::swap(sp.pos[first], sp.pos[second]);
}
void SpAction::_neg_swap(SeqPair& sp, SpAction* action)
{
  auto [first, second] = action->index_pair();
  std::swap(sp.neg[first], sp.neg[second]);
};

void SpAction::_double_swap(SeqPair& sp, SpAction* action)
{
  auto [first1, second1] = action->index_pair();
  std::swap(sp.pos[first1], sp.pos[second1]);
  auto [first2, second2] = action->index_pair();
  std::swap(sp.neg[first2], sp.neg[second2]);
};

void SpAction::_pos_insert(SeqPair& sp, SpAction* action)
{
  auto [first, second] = action->index_pair();
  size_t val = sp.pos[first];
  sp.pos.erase(std::begin(sp.pos) + first);
  sp.pos.insert(std::begin(sp.pos) + second, val);
};

void SpAction::_neg_insert(SeqPair& sp, SpAction* action)
{
  auto [first, second] = action->index_pair();
  size_t val = sp.neg[first];
  sp.neg.erase(std::begin(sp.neg) + first);
  sp.neg.insert(std::begin(sp.neg) + second, val);
};

bool SpPlot::operator()(const SeqPair& sp, std::string filename)
{
  auto eval = SpEvaluate(_netlist);
  eval(sp);
  auto lx = eval.get_lx();
  auto ly = eval.get_ly();
  PyPlot<int64_t> plt;
  for (size_t i = _netlist.num_cells; i < _netlist.num_cells + _netlist.num_clusters; i++) {
    plt.addCluster(lx[i], ly[i], _netlist.dx[i], _netlist.dy[i]);
  }
  for (size_t i = _netlist.num_clusters; i < _netlist.num_macros + _netlist.num_clusters; i++) {
    plt.addMacro(lx[i], ly[i], _netlist.dx[i], _netlist.dy[i]);
  }
  // size_t space = std::max(size_t(_netlist.net_span.size() / 500), size_t(1));
  // for (size_t i = 0; i < _netlist.net_span.size() - 1; i += space) {
  //   double x_0 = std::numeric_limits<double>::max(), y_0 = 0.;
  //   double x_1 = 0., y_1 = std::numeric_limits<double>::max();
  //   double x_2 = std::numeric_limits<double>::lowest(), y_2 = 0.;
  //   double x_3 = 0., y_3 = std::numeric_limits<double>::lowest();
  //   for (size_t j = _netlist.net_span[i]; j < _netlist.net_span[i + 1]; j++) {
  //     double x = eval.get_pin_x().at(j);
  //     double y = eval.get_pin_y().at(j);
  //     if (x < x_0) {
  //       x_0 = x;
  //       y_0 = y;
  //     }
  //     if (y < y_1) {
  //       x_1 = x;
  //       y_1 = y;
  //     }
  //     if (x > x_2) {
  //       x_2 = x;
  //       y_2 = y;
  //     }
  //     if (y > y_3) {
  //       x_3 = x;
  //       y_3 = y;
  //     }
  //   }
  //   // plt.addFlyLine(x_0, y_0, x_1, y_1, x_2, y_2, x_3, y_3);
  // }

  plt.addRectangle(_netlist.region_lx, _netlist.region_ly, _netlist.region_dx, _netlist.region_dy);
  plt.addTitle("Wirelength: " + std::to_string(eval.get_wl()));
  plt.set_limitation(_xlim, _ylim);
  return plt.save(filename);
}
}  // namespace imp

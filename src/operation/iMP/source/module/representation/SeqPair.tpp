#include <algorithm>
#include <ranges>

#include "../../utility/logger/Logger.hpp"
#include "SeqPair.hh"
namespace imp {
template <typename CoordType>
SeqPair<CoordType>::SeqPair(size_t size) : _size(size)
{
  _pos.resize(_size);
  _neg.resize(_size);
  // status.resize(sz);
  std::iota(_pos.begin(), _pos.end(), 0);
  std::iota(_neg.begin(), _neg.end(), 0);
  _initRandom();
}

template <typename CoordType>
SeqPair<CoordType>::SeqPair(const SeqPair<CoordType>& other) : _size(other._size), _pos(other._pos), _neg(other._neg)
{
  // max_status = other.max_status;
  _initRandom();
  // status = other.status;
}

template <typename CoordType>
SeqPair<CoordType>& SeqPair<CoordType>::operator=(const SeqPair<CoordType>& other)
{
  if (_size == other._size) {
    std::copy(other._pos.begin(), other._pos.end(), _pos.begin());
    std::copy(other._neg.begin(), other._neg.end(), _neg.begin());
    // std::copy(other.status.begin(), other.status.end(), status.begin());
  } else {
    _pos = other._pos;
    _neg = other._neg;
    // status = other.status;
    _size = other._size;
  }
  // max_status = other.max_status;
  return *this;
}

template <typename CoordType>
std::pair<CoordType, CoordType> SeqPair<CoordType>::SpLocation::operator()(
    const SeqPair<CoordType>& sp, const std::vector<CoordType>& width, const std::vector<CoordType>& height,
    const std::vector<CoordType>& halo_x, const std::vector<CoordType>& halo_y, std::vector<CoordType>& x, std::vector<CoordType>& y,
    CoordType region_lx, CoordType region_ly, bool is_left, bool is_bottom)
{
  size_t sp_size = sp.get_size();
  if (width.size() != height.size() || width.size() < sp_size || height.size() < sp_size) {
    ERROR("length of width vector && height vector < seqpair-size...");
  }
  if (x.size() <= sp_size) {
    x.resize(width.size());
  }
  if (y.size() <= sp_size) {
    y.resize(height.size());
  }
  CoordType bound_x{0};
  CoordType bound_y{0};

  auto& sp_pos = sp.get_pos();
  auto& sp_neg = sp.get_neg();
  if (sp_size > _match.size()) {
    _match.resize(sp_size);
    _reverse_pos.resize(sp_size);
    _reverse_neg.resize(sp_size);
  }
  std::fill(_match.begin(), _match.end(), 0);
  if (is_left) {
    bound_x = pack(sp_pos, sp_neg, width, halo_x, x);
  } else {
    std::reverse_copy(std::begin(sp_pos), std::end(sp_pos), std::begin(_reverse_pos));
    std::reverse_copy(std::begin(sp_neg), std::end(sp_neg), std::begin(_reverse_neg));
    bound_x = pack(_reverse_pos, _reverse_neg, width, halo_x, x);
    for (size_t i = 0; i < sp_size; i++) {
      x[i] = bound_x - x[i] - width[i] - halo_x[i] + region_lx;
    }
  }
  std::fill(_match.begin(), _match.end(), 0);
  if (is_bottom) {
    if (is_left)
      std::reverse_copy(std::begin(sp_pos), std::end(sp_pos), std::begin(_reverse_pos));
    bound_y = pack(_reverse_pos, sp_neg, height, halo_y, y);
  } else {
    if (is_left)
      std::reverse_copy(std::begin(sp_neg), std::end(sp_neg), std::begin(_reverse_neg));
    bound_y = pack(sp_pos, _reverse_neg, height, halo_y, y);
    for (size_t i = 0; i < sp_size; i++) {
      y[i] = bound_y - y[i] - height[i] - halo_y[i] + region_ly;
    }
  }
  return std::make_pair(bound_x, bound_y);
}

template <typename CoordType>
inline CoordType SeqPair<CoordType>::SpLocation::find(size_t id)
{
  CoordType loc{0};
  auto iter = _bst.lower_bound(id);
  if (iter != _bst.begin()) {
    iter--;
    loc = (*iter).second;
  } else
    loc = 0;
  return loc;
}

template <typename CoordType>
inline void SeqPair<CoordType>::SpLocation::remove(size_t id, CoordType length)
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

template <typename CoordType>
inline CoordType SeqPair<CoordType>::SpLocation::pack(const std::vector<size_t>& pos, const std::vector<size_t>& neg,
                                                      const std::vector<CoordType>& weight, const std::vector<CoordType>& halo,
                                                      std::vector<CoordType>& loc)
{
  _bst.clear();
  _bst[0] = 0;
  size_t size = pos.size();
  for (size_t i = 0; i < size; ++i) {
    _match[neg[i]] = i;
  }
  CoordType t{0};
  for (size_t i = 0; i < size; ++i) {
    size_t p = _match[pos[i]];
    loc[pos[i]] = find(p);
    t = loc[pos[i]] + weight[pos[i]] + 2 * halo[pos[i]];
    _bst[p] = t;
    remove(p, t);
  }
  CoordType length = find(size);
  return length;
}

}  // namespace imp

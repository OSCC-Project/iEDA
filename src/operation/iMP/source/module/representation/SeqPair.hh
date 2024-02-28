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
#pragma once
#include <concepts>
#include <functional>
#include <map>
#include <random>
#include <unordered_map>
#include <vector>
namespace imp {

template <typename Property>
struct SeqPair
{
  template <typename RandGenerator>
  SeqPair(const std::vector<Property>& properties, RandGenerator&);
  SeqPair(const std::vector<Property>& properties);
  ~SeqPair() = default;
  size_t size{0};
  std::vector<size_t> pos{};
  std::vector<size_t> neg{};
  std::vector<Property> properties{};
};
template <typename Property>
template <typename RandGenerator>
SeqPair<Property>::SeqPair(const std::vector<Property>& properties_t, RandGenerator& gen) : SeqPair(properties_t)
{
  // SeqPair::SeqPair(properties_t);
  std::shuffle(std::begin(pos), std::end(pos), gen);
  std::shuffle(std::begin(neg), std::end(neg), gen);
}

template <typename Property>
SeqPair<Property>::SeqPair(const std::vector<Property>& properties_t) : size(properties_t.size()), properties(properties_t)
{
  pos.resize(size);
  neg.resize(size);
  std::iota(pos.begin(), pos.end(), 0);
  std::iota(neg.begin(), neg.end(), 0);
}

template <typename Property, typename Product>
struct FastPackSP
{
  using T = decltype(Product::width);
  using DimFunc = std::function<T(size_t id, const Property&)>;        // Function type of corresponding Width and Height
  using IgnoreFunc = std::function<bool(size_t id, const Property&)>;  // Function type to determine whether to ignore
  void operator()(const SeqPair<Property>& sp, Product& product)
  {
    auto [w, h] = this->operator()(sp, product.x, product.y);
    product.width = w;
    product.height = h;
  }
  std::pair<T, T> operator()(const SeqPair<Property>& sp, std::vector<T>& x, std::vector<T>& y, bool is_left = true, bool is_bottom = true);

  FastPackSP(
      T outline_lx, T outline_ly, DimFunc get_width_t, DimFunc get_height_t,
      IgnoreFunc is_ignore_t = [](size_t, const Property&) { return false; })
      : _outline_lx(outline_lx), _outline_ly(outline_ly), get_width(get_width_t), get_height(get_height_t), is_ignore(is_ignore_t)
  {
  }

 private:
  T find(size_t id);
  void remove(size_t id, T);
  T pack(const std::vector<size_t>& pos, const std::vector<size_t>& neg, const std::vector<Property>& properties, const DimFunc& weight,
         std::vector<T>& loc);
  T _outline_lx;
  T _outline_ly;
  DimFunc get_width;
  DimFunc get_height;
  IgnoreFunc is_ignore;
  std::map<size_t, T> _bst;
  std::vector<size_t> _reverse_pos;
  std::vector<size_t> _reverse_neg;
  std::vector<size_t> _match;
};

template <typename Property, typename Product>
struct FastPackSPWithShape : public FastPackSP<Property, Product>
{
  using T = decltype(Product::width);
  using DimFunc = std::function<T(size_t id, const Property&)>;        // Function type of corresponding Width and Height
  using IgnoreFunc = std::function<bool(size_t id, const Property&)>;  // Function type to determine whether to ignore
  FastPackSPWithShape(
      T outline_lx, T outline_ly, DimFunc get_width_t, DimFunc get_height_t,
      IgnoreFunc is_ignore_t = [](size_t, const Property&) { return false; })
      : FastPackSP<Property, Product>(outline_lx, outline_ly, get_width_t, get_height_t, is_ignore_t)
  {
  }
  void operator()(const SeqPair<Property>& sp, Product& product)
  {
    FastPackSP<Property, Product>::operator()(sp, product);
    if (product.dx.size() != sp.properties.size()) {
      product.dx.resize(sp.properties.size());
    }
    if (product.dy.size() != sp.properties.size()) {
      product.dy.resize(sp.properties.size());
    }
    for (size_t i = 0; i < sp.properties.size(); ++i) {
      product.dx[i] = sp.properties[i].width;
      product.dy[i] = sp.properties[i].height;
    }
  }
};

template <typename Property, typename Product>
inline std::pair<typename FastPackSP<Property, Product>::T, typename FastPackSP<Property, Product>::T>
FastPackSP<Property, Product>::operator()(const SeqPair<Property>& sp, std::vector<T>& x, std::vector<T>& y, bool is_left, bool is_bottom)
{
  T outline_w{0};
  T outline_h{0};
  if (sp.size > _match.size()) {
    _match.resize(sp.size);
    _reverse_pos.resize(sp.size);
    _reverse_neg.resize(sp.size);
  }
  std::unordered_map<size_t, std::pair<T, T>> mask_xy;
  for (size_t i = 0; i < sp.size; i++) {
    if (!is_ignore(i, sp.properties[i]))
      continue;
    mask_xy[i] = std::make_pair(x[i], y[i]);
  }

  std::fill(_match.begin(), _match.end(), 0);
  if (is_left) {
    outline_w = pack(sp.pos, sp.neg, sp.properties, get_width, x);
  } else {
    std::reverse_copy(std::begin(sp.pos), std::end(sp.pos), std::begin(_reverse_pos));
    std::reverse_copy(std::begin(sp.neg), std::end(sp.neg), std::begin(_reverse_neg));
    outline_w = pack(_reverse_pos, _reverse_neg, sp.properties, get_width, x);
    for (size_t i = 0; i < sp.size; i++) {
      x[i] = outline_w - x[i] - get_width(i, sp.properties[i]);
    }
  }
  std::fill(_match.begin(), _match.end(), 0);
  if (is_bottom) {
    if (is_left)
      std::reverse_copy(std::begin(sp.pos), std::end(sp.pos), std::begin(_reverse_pos));
    outline_h = pack(_reverse_pos, sp.neg, sp.properties, get_height, y);
  } else {
    if (is_left)
      std::reverse_copy(std::begin(sp.neg), std::end(sp.neg), std::begin(_reverse_neg));
    outline_h = pack(sp.pos, _reverse_neg, sp.properties, get_height, y);
    for (size_t i = 0; i < sp.size; i++) {
      y[i] = outline_h - y[i] - get_height(i, sp.properties[i]);
    }
  }
  for (size_t i = 0; i < sp.size; i++) {
    if (is_ignore(i, sp.properties[i])) {
      auto&& [xx, yy] = mask_xy[i];
      x[i] = xx;
      y[i] = yy;
    } else {
      x[i] += _outline_lx;
      y[i] += _outline_ly;
    }
  }

  return {outline_w, outline_h};
}

template <typename Property, typename Product>
inline typename FastPackSP<Property, Product>::T FastPackSP<Property, Product>::find(size_t id)
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

template <typename Property, typename Product>
void FastPackSP<Property, Product>::remove(size_t id, T length)
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

template <typename Property, typename Product>
typename FastPackSP<Property, Product>::T FastPackSP<Property, Product>::pack(const std::vector<size_t>& pos,
                                                                              const std::vector<size_t>& neg,
                                                                              const std::vector<Property>& properties,
                                                                              const DimFunc& weight, std::vector<T>& loc)
{
  _bst.clear();
  _bst[0] = 0;
  for (size_t i = 0; size_t id : neg) {
    _match[id] = i++;
  }
  T t{0};
  for (size_t id : pos) {
    if (is_ignore(id, properties[id]))
      continue;
    size_t p = _match[id];
    loc[id] = find(p);
    t = loc[id] + weight(id, properties[id]);
    _bst[p] = t;
    remove(p, t);
  }
  T length = find(pos.size());
  return length;
}

template <typename RandGenerator>
std::pair<size_t, size_t> rand_index_pair(size_t idx_begin, size_t idx_end, RandGenerator& gen)
{
  std::uniform_int_distribution<size_t> get_random_index(idx_begin, idx_end);
  size_t first = get_random_index(gen);
  size_t second = get_random_index(gen);
  while (first == second) {
    second = get_random_index(gen);
  }
  return {first, second};
}
struct PosSwap
{
  template <typename Property, typename RandGenerator>
  void operator()(SeqPair<Property>& sp, RandGenerator& gen)
  {
    auto [first, second] = rand_index_pair(0, sp.size - 1, gen);
    std::swap(sp.pos[first], sp.pos[second]);
  }
};

struct NegSwap
{
  template <typename Property, typename RandGenerator>
  void operator()(SeqPair<Property>& sp, RandGenerator& gen)
  {
    auto [first, second] = rand_index_pair(0, sp.size - 1, gen);
    std::swap(sp.neg[first], sp.neg[second]);
  }
};

struct DoubleSwap
{
  template <typename Property, typename RandGenerator>
  void operator()(SeqPair<Property>& sp, RandGenerator& gen)
  {
    auto [first1, second1] = rand_index_pair(0, sp.size - 1, gen);
    std::swap(sp.pos[first1], sp.pos[second1]);
    auto [first2, second2] = rand_index_pair(0, sp.size - 1, gen);
    std::swap(sp.neg[first2], sp.neg[second2]);
  }
};
struct PosInsert
{
  template <typename Property, typename RandGenerator>
  void operator()(SeqPair<Property>& sp, RandGenerator& gen)
  {
    auto [first, second] = rand_index_pair(0, sp.size - 1, gen);
    size_t val = sp.pos[first];
    sp.pos.erase(std::begin(sp.pos) + first);
    sp.pos.insert(std::begin(sp.pos) + second, val);
  }
};

struct NegInsert
{
  template <typename Property, typename RandGenerator>
  void operator()(SeqPair<Property>& sp, RandGenerator& gen)
  {
    auto [first, second] = rand_index_pair(0, sp.size - 1, gen);
    size_t val = sp.neg[first];
    sp.neg.erase(std::begin(sp.neg) + first);
    sp.neg.insert(std::begin(sp.neg) + second, val);
  }
};

struct Resize
{
  template <typename Property, typename RandGenerator>
  void operator()(SeqPair<Property>& sp, RandGenerator& gen)
  {
    std::uniform_int_distribution<size_t> get_random_index(0, sp.size - 1);
    size_t try_times = 20;
    bool success_flag;
    for (size_t i = 0; i < try_times; ++i) {
      size_t idx = get_random_index(gen);
      success_flag = sp.properties[idx].resize(gen);
      if (success_flag == true) {
        break;
      }
    }
  }
};

}  // namespace imp
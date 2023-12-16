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
#include <any>
#include <cstdint>
#include <functional>
#include <map>
#include <random>
#include <span>
#include <tuple>
#include <vector>

namespace imp {

template <typename CoordType>
class SeqPair
{
 public:
  static_assert(std::is_arithmetic<CoordType>::value, "SeqPair requaires a numeric type.");
  template <typename U>
  friend struct SpAction;
  SeqPair(const size_t size);
  SeqPair(const SeqPair& other);
  ~SeqPair() = default;
  // SeqPair& operator=(const SeqPair& other);
  SeqPair<CoordType>& operator=(const SeqPair<CoordType>& other);
  bool operator==(const SeqPair& other)
  {
    if (_size != other._size)
      return false;
    for (size_t i = 0; i < _size; ++i) {
      if (_pos[i] != other._pos[i] || _neg[i] != other._neg[i]) {
        return false;
      }
    }
    return true;
  }
  // getter
  size_t get_size() const { return _size; }
  const std::vector<size_t>& get_pos() const { return _pos; }
  const std::vector<size_t>& get_neg() const { return _neg; }

  std::pair<CoordType, CoordType> packing(const std::vector<CoordType>& dx, const std::vector<CoordType>& dy,
                                          const std::vector<CoordType>& halo_x, const std::vector<CoordType>& halo_y,
                                          std::vector<CoordType>& lx, std::vector<CoordType>& ly, CoordType region_lx, CoordType region_ly,
                                          bool is_left = true, bool is_bottom = true)
  /*packing seqence-pair to two-dimensional coords*/
  {
    return _get_location(*this, dx, dy, halo_x, halo_y, lx, ly, region_lx, region_ly, is_left, is_bottom);
  }
  void randomize()
  {
    std::shuffle(std::begin(_pos), std::end(_pos), _gen);
    std::shuffle(std::begin(_neg), std::end(_neg), _gen);
  }

  void pos_swap(std::pair<size_t, size_t> pos_index_pair) { std::swap(_pos[pos_index_pair.first], _pos[pos_index_pair.second]); }

  void neg_swap(std::pair<size_t, size_t> neg_index_pair) { std::swap(_neg[neg_index_pair.first], _neg[neg_index_pair.second]); };

  void double_swap(std::pair<size_t, size_t> pos_index_pair, std::pair<size_t, size_t> neg_index_pair)
  {
    pos_swap(pos_index_pair);
    neg_swap(neg_index_pair);
  };

  void pos_insert(std::pair<size_t, size_t> pos_index_pair)
  {
    size_t val = _pos[pos_index_pair.first];
    _pos.erase(std::begin(_pos) + pos_index_pair.first);
    _pos.insert(std::begin(_pos) + pos_index_pair.second, val);
  };

  void neg_insert(std::pair<size_t, size_t> neg_index_pair)
  {
    size_t val = _neg[neg_index_pair.first];
    _neg.erase(std::begin(_neg) + neg_index_pair.first);
    _neg.insert(std::begin(_neg) + neg_index_pair.second, val);
  };

  struct SpLocation
  {
   public:
    std::pair<CoordType, CoordType> operator()(const SeqPair<CoordType>& sp, const std::vector<CoordType>& width,
                                               const std::vector<CoordType>& height, const std::vector<CoordType>& halo_x,
                                               const std::vector<CoordType>& halo_y, std::vector<CoordType>& x, std::vector<CoordType>& y,
                                               CoordType region_lx, CoordType region_ly, bool is_left = true, bool is_bottom = true);

   private:
    CoordType find(size_t id);
    void remove(size_t id, CoordType);
    CoordType pack(const std::vector<size_t>& pos, const std::vector<size_t>& neg, const std::vector<CoordType>& weight,
                   const std::vector<CoordType>& halo, std::vector<CoordType>& loc);
    std::map<size_t, CoordType> _bst;
    std::vector<size_t> _reverse_pos{};
    std::vector<size_t> _reverse_neg{};
    std::vector<size_t> _match{};
    // size_t _size;
  };
  void randomDisturb()
  {
    std::pair<size_t, size_t> index_pair = _makeRandomIndexPair();
    auto disturb_func_index = _random_probability(_gen);
    switch (disturb_func_index) {
      case 0:
        double_swap(index_pair, _makeRandomIndexPair());
        break;
      case 1:
        pos_swap(index_pair);
        break;
      case 2:
        neg_swap(index_pair);
        break;
      case 3:
        pos_insert(index_pair);
        break;
      case 4:
        neg_insert(index_pair);
        break;
    }
  }

  void setDisturbProb(int prob_double_swap, int prob_pos_swap, int prob_neg_swap, int prob_pos_insert, int prob_neg_insert)
  {
    _random_probability
        = std::discrete_distribution<size_t>({prob_double_swap, prob_pos_swap, prob_neg_swap, prob_pos_insert, prob_neg_insert});
  }

 private:
  SpLocation _get_location{};
  size_t _size{0};
  std::vector<size_t> _pos{};
  std::vector<size_t> _neg{};
  std::mt19937 _gen;
  std::uniform_int_distribution<size_t> _random_index;
  std::discrete_distribution<size_t> _random_probability;

  std::pair<size_t, size_t> _makeRandomIndexPair()
  {
    size_t first = _random_index(_gen);
    size_t second = _random_index(_gen);
    while (first == second) {
      second = _random_index(_gen);
    }
    return std::make_pair(first, second);
  }

  void _initRandom()
  {
    std::random_device rd;
    _gen = std::mt19937(rd());
    _random_index = std::uniform_int_distribution<size_t>(size_t(0), size_t(_size - 1));
    _random_probability = std::discrete_distribution<size_t>({150, 150, 80, 100, 100});
  }
};

template <typename CoordType>
auto makeRandomSeqPair(size_t sz) -> SeqPair<CoordType>
{
  SeqPair<CoordType> sp(sz);
  sp.randomize();
  return sp;
}

}  // namespace imp
#include "SeqPair.tpp"
#endif
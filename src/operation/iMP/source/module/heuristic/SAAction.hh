/**
 * @file SAAction.hh
 * @author Yuezuo Liu (yuezuoliu@163.com)
 * @brief
 * @version 0.1
 * @date 2023-12-1
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef IMP_SAACTION_H
#define IMP_SAACTION_H
#include <cstdint>
#include <functional>
#include <random>
#include <vector>

#include "Logger.hpp"
#include "SAPlacement.hh"

namespace imp {

template <typename CoordType, typename RepresentType>
struct SAAction
{
  void operator()(SAPlacement<CoordType, RepresentType>& placement) { randomActionUntilSucceed(placement); }
  SAAction() { _initRandom(); }
  SAAction(const SAAction& other) { _initRandom(); }
  ~SAAction() = default;

  // void setActionProb(size_t prob_change_represent, size_t prob_change_shape)
  // {
  //   _action_probability = std::discrete_distribution<size_t>({prob_change_represent, prob_change_shape});
  // }

  bool randomAction(SAPlacement<CoordType, RepresentType>& placement)
  {
    size_t rand = _random_index(_gen);
    if (rand == 0) {
      return _changeRepresent(placement);
    } else {
      return _changeShape(placement);
    }
  }

  void randomActionUntilSucceed(SAPlacement<CoordType, RepresentType>& placement)
  {
    bool success = randomAction(placement);
    while (!success) {
      success = randomAction(placement);
    }
  }

 private:
  void _initRandom()
  {
    std::random_device rd;
    _gen = std::mt19937(rd());
    _random_index = std::uniform_int_distribution<size_t>(0, 1);
    // _action_probability = std::discrete_distribution<size_t>({100, 100});
  }

  bool _changeRepresent(SAPlacement<CoordType, RepresentType>& placement)
  {
    placement._representation.randomDisturb();
    return true;
  }
  bool _changeShape(SAPlacement<CoordType, RepresentType>& placement) { return false; }

  std::mt19937 _gen;
  std::uniform_int_distribution<size_t> _random_index;
  // std::discrete_distribution<size_t> _action_probability;
};

}  // namespace imp

#endif
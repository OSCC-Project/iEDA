#pragma once

#include "LayerCoord.hpp"
#include "RTU.hpp"

namespace irt {

template <typename T>
class Segment
{
 public:
  Segment() = default;
  Segment(const T& first, const T& second)
  {
    _first = first;
    _second = second;
  }
  ~Segment() = default;
  // getter
  T& get_first() { return _first; }
  T& get_second() { return _second; }
  const T& get_first() const { return _first; }
  const T& get_second() const { return _second; }
  // setter
  void set_first(const T& x) { _first = x; }
  void set_second(const T& y) { _second = y; }
  // function

 private:
  T _first;
  T _second;
};

struct SortSegmentInnerXASC
{
  void operator()(Segment<LayerCoord>& a) const
  {
    LayerCoord& first_coord = a.get_first();
    LayerCoord& second_coord = a.get_second();
    if (CmpLayerCoordByXASC()(first_coord, second_coord)) {
      return;
    }
    std::swap(first_coord, second_coord);
  }
};

struct SortSegmentInnerYASC
{
  void operator()(Segment<LayerCoord>& a) const
  {
    LayerCoord& first_coord = a.get_first();
    LayerCoord& second_coord = a.get_second();
    if (CmpLayerCoordByYASC()(first_coord, second_coord)) {
      return;
    }
    std::swap(first_coord, second_coord);
  }
};

struct SortSegmentInnerLayerASC
{
  void operator()(Segment<LayerCoord>& a) const
  {
    LayerCoord& first_coord = a.get_first();
    LayerCoord& second_coord = a.get_second();
    if (CmpLayerCoordByLayerASC()(first_coord, second_coord)) {
      return;
    }
    std::swap(first_coord, second_coord);
  }
};

struct CmpSegmentXASC
{
  bool operator()(Segment<LayerCoord>& a, Segment<LayerCoord>& b) const
  {
    if (a.get_first() != b.get_first()) {
      return CmpLayerCoordByXASC()(a.get_first(), b.get_first());
    } else {
      return CmpLayerCoordByXASC()(a.get_second(), b.get_second());
    }
  }
};

struct CmpSegmentYASC
{
  bool operator()(Segment<LayerCoord>& a, Segment<LayerCoord>& b) const
  {
    if (a.get_first() != b.get_first()) {
      return CmpLayerCoordByYASC()(a.get_first(), b.get_first());
    } else {
      return CmpLayerCoordByYASC()(a.get_second(), b.get_second());
    }
  }
};

struct CmpSegmentLayerASC
{
  bool operator()(Segment<LayerCoord>& a, Segment<LayerCoord>& b) const
  {
    if (a.get_first() != b.get_first()) {
      return CmpLayerCoordByLayerASC()(a.get_first(), b.get_first());
    } else {
      return CmpLayerCoordByLayerASC()(a.get_second(), b.get_second());
    }
  }
};

}  // namespace irt

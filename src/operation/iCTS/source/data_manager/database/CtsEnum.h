#pragma once

#include <cassert>

#include "pgl.h"

namespace icts {

enum ConceptEnum {
  kPoint = 0,
  kPin = 1,
  kInstance = 2,
  kNet = 3,
  kWire = 4,
  kNone = 5
};

enum OrientationEnum { kHorizontal = 0, kVertical = 1 };

enum DirectionEnum { kWEST = 0, kEAST = 1, kSOUTH = 2, kNORTH = 3 };

enum SegmentType { kMANHATAN_ARC, kRECTILINEAR };

class Concept {
 public:
  Concept() : _val(kNone) {}
  Concept(const ConceptEnum cpt) : _val(cpt) {}
  Concept(const Concept &that) : _val(that._val) {}
  ~Concept() = default;
  Concept &operator=(const Concept &that) {
    _val = that.to_int();
    return *this;
  }
  bool operator==(const Concept &that) const { return _val == that.to_int(); }
  int to_int() const { return _val; }

 private:
  int _val;
};

class Direction {
 public:
  Direction() : _val(kWEST) {}
  Direction(DirectionEnum dir) : _val(dir) {}
  Direction(const Direction &that) : _val(that._val) {}
  Direction &operator=(DirectionEnum dir) {
    _val = dir;
    return *this;
  }
  Direction &operator=(Direction dir) {
    _val = dir._val;
    return *this;
  }
  Direction backward() {
    _val ^= 1;
    return *this;
  }
  gtl::orientation_2d to_orientation() const {
    return gtl::orientation_2d(_val > 1 ? gtl::HORIZONTAL : gtl::VERTICAL);
  }
  int to_int() const { return _val; }

 private:
  int _val;
};

}  // namespace icts
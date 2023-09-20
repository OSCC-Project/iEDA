#pragma once

#include <iostream>
#include <memory>
#include <vector>

#include "salt/utils/utils.h"

// #define DTYPE int  // same as flute.h, will overwrite it
typedef int DTYPE;

namespace salt {
using namespace std;
using Point = utils::PointT<DTYPE>;
using Box = utils::BoxT<DTYPE>;

class Pin
{
 public:
  Point loc;
  int id;  // 0 for source, should be in range [0, pin_num) for a net
  double cap;

  Pin(const Point& l, int i = -1, double c = 0.0) : loc(l), id(i), cap(c) {}
  Pin(DTYPE x, DTYPE y, int i = -1, double c = 0.0) : loc(x, y), id(i), cap(c) {}

  bool IsSource() const { return id == 0; }
  bool IsSink() const { return !IsSource(); }
};

class Net
{
 public:
  int id;
  string name;
  bool with_cap = false;

  vector<shared_ptr<Pin>> pins;  // source is always the first one

  void init(const int& id, const string& name, const vector<shared_ptr<Pin>>& pins)
  {
    this->id = id;
    this->name = name;
    this->pins = pins;
    this->with_cap = true;
  }

  void RanInit(int i, int numPin, DTYPE width = 100, DTYPE height = 100);  // random initialization

  shared_ptr<Pin> source() const { return pins[0]; }

  // File read/write
  // ------
  // Format:
  // Net <net_id> <net_name> <pin_num> [-cap]
  // 0 x0 y0 [cap0]
  // 1 x1 y1 [cap1]
  // ...
  // ------
  bool Read(istream& is);
  void Read(const string& fileName);
  string GetHeader() const;
  void Write(ostream& os) const;
  void Write(const string& prefix, bool withNetInfo = true) const;

  friend ostream& operator<<(ostream& os, const Net& net)
  {
    net.Write(os);
    return os;
  }
};

}  // namespace salt
#pragma once

#include "GdsElement.hpp"
#include "GdsStrans.hpp"

namespace idb {

class GdsAref : public GdsElemBase
{
 public:
  GdsAref() : GdsElemBase(GdsElemType::kAref), sname(), strans(), col(1), row(1) {}

  GdsAref& operator=(const GdsAref& rhs)
  {
    GdsElemBase::operator=(rhs);
    sname = rhs.sname;
    strans = rhs.strans;
    col = rhs.col;
    row = rhs.row;

    return *this;
  }

  void reset() override
  {
    reset_base();
    sname.clear();
    strans.reset();
    col = 1;
    row = 1;
  }

  // members
  std::string sname;
  GdsStrans strans;
  int16_t col;
  int16_t row;
};

}  // namespace idb

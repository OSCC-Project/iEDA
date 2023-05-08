#pragma once

#include <string>

#include "GdsElement.hpp"
#include "GdsStrans.hpp"

namespace idb {

class GdsSref : public GdsElemBase
{
 public:
  GdsSref() : GdsElemBase(GdsElemType::kSref), sname(), strans() {}

  GdsSref& operator=(const GdsSref& rhs)
  {
    GdsElemBase::operator=(rhs);
    sname = rhs.sname;
    strans = rhs.strans;

    return *this;
  }

  void reset() override
  {
    reset_base();
    sname.clear();
    strans.reset();
  }

  // members
  std::string sname;  // structure name
  GdsStrans strans;
};

}  // namespace idb

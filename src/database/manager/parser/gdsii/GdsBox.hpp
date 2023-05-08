#pragma once

#include "GdsElement.hpp"

namespace idb {

class GdsBox : public GdsElemBase
{
 public:
  GdsBox() : GdsElemBase(GdsElemType::kBox), layer(0), box_type(0) {}

  GdsBox& operator=(const GdsBox& rhs)
  {
    GdsElemBase::operator=(rhs);
    layer = rhs.layer;
    box_type = rhs.box_type;

    return *this;
  }

  void reset() override
  {
    reset_base();
    layer = 0;
    box_type = 0;
  }

  // members
  GdsLayer layer;
  GdsBoxType box_type;
};

}  // namespace idb

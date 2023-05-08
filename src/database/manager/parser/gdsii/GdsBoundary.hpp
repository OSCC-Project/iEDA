#pragma once

#include "GdsElement.hpp"

namespace idb {

class GdsBoundary : public GdsElemBase
{
 public:
  GdsBoundary() : GdsElemBase(GdsElemType::kBoundary), layer(0), data_type(0) {}

  GdsBoundary& operator=(const GdsBoundary& rhs)
  {
    GdsElemBase::operator=(rhs);
    layer = rhs.layer;
    data_type = rhs.data_type;

    return *this;
  }

  void reset() override
  {
    reset_base();
    layer = 0;
    data_type = 0;
  }

  // members
  GdsLayer layer;
  GdsDataType data_type;
};

}  // namespace idb

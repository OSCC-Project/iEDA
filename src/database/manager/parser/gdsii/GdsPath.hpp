#pragma once

#include "GdsElement.hpp"

namespace idb {

class GdsPath : public GdsElemBase
{
 public:
  GdsPath() : GdsElemBase(GdsElemType::kPath), layer(0), data_type(0), path_type(GdsPathType::kDefault), width(0) {}

  GdsPath& operator=(const GdsPath& rhs)
  {
    GdsElemBase::operator=(rhs);
    layer = rhs.layer;
    data_type = rhs.data_type;
    path_type = rhs.path_type;
    width = rhs.width;

    return *this;
  }

  void reset() override
  {
    reset_base();
    layer = 0;
    data_type = 0;
    path_type = GdsPathType::kDefault;
    width = 0;
  }

  // members
  GdsLayer layer;
  GdsDataType data_type;
  GdsPathType path_type;
  GdsWidth width;
};

}  // namespace idb

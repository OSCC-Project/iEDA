#pragma once

#include "GdsElement.hpp"

namespace idb {

class GdsNode : public GdsElemBase
{
 public:
  GdsNode() : GdsElemBase(GdsElemType::kNode), layer(0), node_type(0) {}

  GdsNode& operator=(const GdsNode& rhs)
  {
    GdsElemBase::operator=(rhs);
    layer = rhs.layer;
    node_type = rhs.node_type;

    return *this;
  }

  void reset() override
  {
    reset_base();
    layer = 0;
    node_type = 0;
  }

  // members
  GdsLayer layer;
  GdsNodeType node_type;
};

}  // namespace idb

#pragma once

#include "rt_serialize.hpp"

namespace iplf {
class RtPersister : public Persister<std::vector<irt::Net>>
{
 public:
  RtPersister(const std::string& path);
};

}  // namespace iplf
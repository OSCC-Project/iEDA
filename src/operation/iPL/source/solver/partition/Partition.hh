
#pragma once

#include "MetisParam.hh"

namespace ipl {
class Partition
{
 public:
  Partition(){};
  ~Partition(){};

  virtual void initMetis() = 0; // create MetisParam
  void MetisPartition(MetisParam* metis_param);
};

}  // namespace ipl
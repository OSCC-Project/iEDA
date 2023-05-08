
#pragma once
#include "MPDB.h"
namespace ipl::imp {
class Checker
{
 public:
  Checker(){};
  ~Checker(){};

 private:
  bool checkOverlape(vector<FPInst*> macro_list);
  bool checkOutborder(vector<FPInst*> macro_list, Coordinate* ld, Coordinate* ru);
};

}  // namespace ifp
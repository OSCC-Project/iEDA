#pragma once

#include <string>
#include <vector>

#include "MPDB.hh"
#include "Module.hh"
#include "Setting.hh"
#include "metis.h"

namespace ipl::imp {
class HierParttion
{
 public:
  HierParttion(MPDB* mdb, Setting* set) : _mdb(mdb), _set(set){};
  ~HierParttion(){};
  void init();

 private:
  MPDB* _mdb;
  Setting* _set;
  Module* _top_module;
};
}  // namespace ipl::imp
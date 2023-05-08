
#pragma once

#include "FPRect.hh"

namespace ipl::imp {

class FPLayout
{
 public:
  FPLayout(){};
  ~FPLayout(){};

  // getter
  FPRect* get_die_shape() const { return _die_shape; }
  FPRect* get_core_shape() const { return _core_shape; }

  // setter
  void set_die_shape(FPRect* die) { _die_shape = die; }
  void set_core_shape(FPRect* core) { _core_shape = core; }

 private:
  FPRect* _die_shape;
  FPRect* _core_shape;
};

}  // namespace ipl::imp

#include "py_inst.h"

#include <idm.h>

namespace python_interface {
bool fpPlaceInst(const std::string& inst_name, int llx, int lly, const std::string& orient, const std::string& cellmaster,
                 const std::string& source)
{
  dmInst->placeInst(inst_name, llx, lly, orient, cellmaster, source);
  return true;
}

}  // namespace python_interface
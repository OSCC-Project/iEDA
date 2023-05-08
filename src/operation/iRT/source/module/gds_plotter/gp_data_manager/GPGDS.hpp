#pragma once

#include "GPStruct.hpp"
#include "MTree.hpp"

namespace irt {

class GPGDS
{
 public:
  GPGDS() = default;
  GPGDS(std::string top_name) { _top_name = top_name; }
  ~GPGDS() = default;
  // getter
  std::string& get_top_name() { return _top_name; }
  std::vector<GPStruct>& get_struct_list() { return _struct_list; }
  // setter

  // function
  void addStruct(GPStruct& gp_struct) { _struct_list.push_back(gp_struct); }

 private:
  std::string _top_name = "top";
  std::vector<GPStruct> _struct_list;
};

}  // namespace irt

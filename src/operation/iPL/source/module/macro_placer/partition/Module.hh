#pragma once
#include <map>
#include <queue>
#include <string>
#include <vector>

#include "database/FPInst.hh"
namespace ipl::imp {

class Module
{
 public:
  Module(){};
  ~Module(){};
  void set_name(std::string name) { _name = name; }
  void add_inst(FPInst* inst, std::queue<std::string> level_name_list = {}, std::string father_name = "top");
  void set_layer(int layer) { _layer = layer; }

  std::string get_name() { return _name; }
  std::vector<FPInst*> get_macro_list() { return _macro_list; }
  std::vector<FPInst*> get_stdcell_list() { return _stdcell_list; }
  std::vector<Module*> get_child_module_list() { return _child_module_list; }
  bool hasChildModule() { return _child_module_list.size() != 0; }
  int get_layer() { return _layer; }
  Module* findChildMoudle(std::string module_name);

 private:
  std::queue<string> split(const string& str);
  std::string _name = "top";
  int _layer = 0;
  std::vector<FPInst*> _macro_list;
  std::vector<FPInst*> _stdcell_list;
  std::vector<Module*> _child_module_list;
  std::map<std::string, Module*> _name_to_module_map;
};

}  // namespace ipl::imp
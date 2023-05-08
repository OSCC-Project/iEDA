#pragma once

#include <map>

#include "PlacerDB.hh"
#include "database/FPDesign.hh"
#include "database/FPLayout.hh"

using std::map;

namespace ipl::imp {

class IPLDBWDatabase
{
 public:
  IPLDBWDatabase(ipl::PlacerDB* db) : _ipl_db(db), _layout(new FPLayout()), _design(new FPDesign()) {}
  ~IPLDBWDatabase();

  // function
  // ipl::Instance* find_ipl_inst(FPInst* imp_inst) { return _ipl_inst_map[imp_inst]; }
  // ipl::Net* find_ipl_net(FPNet* imp_net) { return _ipl_net_map[imp_net]; }
  // ipl::Pin* find_ipl_pin(FPPin* imp_pin) { return _ipl_pin_map[imp_pin]; }
  FPInst* find_imp_inst(ipl::Instance* ipl_inst)
  {
    FPInst* imp_inst = nullptr;
    auto imp_inst_iter = _fp_inst_map.find(ipl_inst);
    if (imp_inst_iter != _fp_inst_map.end()) {
      imp_inst = imp_inst_iter->second;
    }
    return imp_inst;
  }
  // FPNet* find_imp_Net(ipl::Net* ipl_net) { return _fp_net_map[ipl_net]; }
  // FPPin* find_imp_pin(ipl::Pin* ipl_pin) { return _fp_pin_map[ipl_pin]; }

 private:
  ipl::PlacerDB* _ipl_db;

  FPLayout* _layout;
  FPDesign* _design;

  map<ipl::Instance*, FPInst*> _fp_inst_map;
  map<ipl::Pin*, FPPin*> _fp_pin_map;
  map<ipl::Net*, FPNet*> _fp_net_map;

  map<FPInst*, ipl::Instance*> _ipl_inst_map;
  map<FPPin*, ipl::Pin*> _ipl_pin_map;
  map<FPNet*, ipl::Net*> _ipl_net_map;

  friend class IPLDBWrapper;
};

inline IPLDBWDatabase::~IPLDBWDatabase()
{
  delete _layout;
  delete _design;
}

}  // namespace ipl::imp
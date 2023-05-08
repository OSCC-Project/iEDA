#ifndef SRC_EVALUATOR_SOURCE_WRAPPER_DATABASE_EVALDESIGN_HPP_
#define SRC_EVALUATOR_SOURCE_WRAPPER_DATABASE_EVALDESIGN_HPP_

#include <map>
#include <string>
#include <vector>

#include "CongInst.hpp"
#include "CongNet.hpp"
#include "CongPin.hpp"
#include "GDSNet.hpp"
#include "WLNet.hpp"
#include "WLPin.hpp"

namespace eval {

class Design
{
 public:
  Design() = default;
  Design(const Design&) = delete;
  Design(Design&&) = delete;
  ~Design();

  Design& operator=(const Design&) = delete;
  Design& operator=(Design&&) = delete;

  // getter.
  std::string get_design_name() const { return _design_name; }

  std::vector<WLNet*> get_wl_net_list() const { return _wl_net_list; }
  std::vector<WLPin*> get_wl_pin_list() const { return _wl_pin_list; }
  std::map<std::string, WLNet*> get_name_net_map() const { return _name_to_net_map; }

  std::vector<CongInst*> get_cong_inst_list() const { return _cong_inst_list; }
  std::vector<CongNet*> get_cong_net_list() const { return _cong_net_list; }
  std::vector<CongPin*> get_cong_pin_list() const { return _cong_pin_list; }

  std::vector<GDSNet*> get_gds_net_list() const { return _gds_net_list; }

  // setter.
  void set_design_name(std::string design_name) { _design_name = std::move(design_name); }
  void add_net(WLNet* net);
  void add_net(CongNet* net);
  void add_net(GDSNet* net);
  void add_pin(WLPin* pin);
  void add_pin(CongPin* pin);
  void add_instance(CongInst* inst);

  // finder
  WLNet* find_wl_net(const std::string& net_name) const;
  WLPin* find_wl_pin(const std::string& pin_name) const;
  CongInst* find_cong_inst(const std::string& inst_name) const;
  CongNet* find_cong_net(const std::string& net_name) const;
  CongPin* find_cong_pin(const std::string& pin_name) const;

 private:
  std::string _design_name;
  std::vector<WLNet*> _wl_net_list;
  std::vector<WLPin*> _wl_pin_list;
  std::vector<CongInst*> _cong_inst_list;
  std::vector<CongNet*> _cong_net_list;
  std::vector<CongPin*> _cong_pin_list;
  std::vector<GDSNet*> _gds_net_list;

  std::map<std::string, WLNet*> _name_to_net_map;
  std::map<std::string, WLPin*> _name_to_pin_map;
  std::map<std::string, CongInst*> _name_to_inst_map;
  std::map<std::string, CongNet*> _name_to_cong_net_map;
  std::map<std::string, CongPin*> _name_to_cong_pin_map;
};

inline Design::~Design()
{
  for (auto* inst : _cong_inst_list) {
    delete inst;
  }
  for (auto* net : _wl_net_list) {
    delete net;
  }
  _wl_net_list.clear();
  for (auto* pin : _wl_pin_list) {
    delete pin;
  }
  _wl_pin_list.clear();
  _name_to_inst_map.clear();
  _name_to_net_map.clear();
  _name_to_pin_map.clear();
}

inline void Design::add_instance(CongInst* inst)
{
  _cong_inst_list.push_back(inst);
  _name_to_inst_map.emplace(inst->get_name(), inst);
}

inline void Design::add_net(WLNet* net)
{
  _wl_net_list.push_back(net);
  _name_to_net_map.emplace(net->get_name(), net);
}

inline void Design::add_pin(WLPin* pin)
{
  _wl_pin_list.push_back(pin);
  _name_to_pin_map.emplace(pin->get_name(), pin);
}

inline void Design::add_net(CongNet* net)
{
  _cong_net_list.push_back(net);
}

inline void Design::add_pin(CongPin* pin)
{
  _cong_pin_list.push_back(pin);
}

inline void Design::add_net(GDSNet* net)
{
  _gds_net_list.push_back(net);
}

inline WLNet* Design::find_wl_net(const std::string& net_name) const
{
  WLNet* net = nullptr;
  auto net_iter = _name_to_net_map.find(net_name);
  if (net_iter != _name_to_net_map.end()) {
    net = (*net_iter).second;
  }
  return net;
}

inline WLPin* Design::find_wl_pin(const std::string& pin_name) const
{
  WLPin* pin = nullptr;
  auto pin_iter = _name_to_pin_map.find(pin_name);
  if (pin_iter != _name_to_pin_map.end()) {
    pin = (*pin_iter).second;
  }
  return pin;
}

inline CongInst* Design::find_cong_inst(const std::string& inst_name) const
{
  CongInst* inst = nullptr;
  auto inst_iter = _name_to_inst_map.find(inst_name);
  if (inst_iter != _name_to_inst_map.end()) {
    inst = (*inst_iter).second;
  }
  return inst;
}

}  // namespace eval

#endif  // SRC_EVALUATOR_SOURCE_WRAPPER_DATABASE_EVALDESIGN_HPP_

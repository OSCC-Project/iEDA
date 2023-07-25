#include "MP.hh"

#include <unordered_map>

#include "DataManager.hh"
#include "Design.hh"
#include "Logger.hpp"
#include "NetList.hh"
#include "SA.hh"
namespace imp {

MacroPlacer::MacroPlacer(DataManager* dm, Option* opt)
{
}

MacroPlacer::MacroPlacer(const std::string& idb_json, const std::string& opt_json) : MacroPlacer()
{
  setDataManager(idb_json);
}

MacroPlacer::MacroPlacer() : _dm(new DataManager())
{
}

void MacroPlacer::setDataManager(DataManager* dm)
{
  if (_dm != nullptr)
    delete _dm;
  _dm = dm;
}

MacroPlacer::~MacroPlacer()
{
  delete _dm;
}

void MacroPlacer::runMP()
{
  auto netlist = pl_to_netlist();
  netlist.autoCellsClustering();
  for (auto var : netlist.report()) {
    INFO(var);
  }
}

NetList MacroPlacer::pl_to_netlist()
{
  Design* design = _dm->get_design();
  const auto& vertexs = design->get_instance_list();
  std::vector<int64_t> lx;
  std::vector<int64_t> ly;
  std::vector<int64_t> dx;
  std::vector<int64_t> dy;
  std::vector<int64_t> area;
  std::vector<NetList::VertexType> type;

  for (Instance* v : design->get_instance_list()) {
    lx.push_back(v->get_coordi().get_x());
    ly.push_back(v->get_coordi().get_y());
    dx.push_back(v->get_shape_width());
    dy.push_back(v->get_shape_height());
    area.push_back(dx.back() * dy.back());
    if (!v->isFixed()) {
      if ((v->get_cell_master()->get_cell_type()) != CELL_TYPE::kMacro)
        type.push_back(NetList::kStdCell);
      else {
        type.push_back(NetList::kMacro);
      }
    } else {
      type.push_back(NetList::kFixInst);
    }
  }

  std::vector<size_t> net_span;
  std::vector<size_t> pin2vertex;
  std::vector<int64_t> pin_x_off;
  std::vector<int64_t> pin_y_off;
  net_span.push_back(0);
  for (Net* net : design->get_net_list()) {
    for (Pin* pin : net->get_pins()) {
      if (pin->isInstancePort()) {
        Instance* v = pin->get_instance();
        pin2vertex.push_back(v->get_inst_id());
        pin_x_off.push_back(pin->get_offset_coordi().get_x());
        pin_y_off.push_back(pin->get_offset_coordi().get_y());
      } else {
        type.push_back(NetList::kTerminal);
        lx.push_back(pin->get_center_coordi().get_x());
        ly.push_back(pin->get_center_coordi().get_y());
        dx.push_back(0);
        dy.push_back(0);
        area.push_back(0);
        pin_x_off.push_back(0);
        pin_y_off.push_back(0);
        pin2vertex.push_back(type.size() - 1);
      }
    }
    net_span.push_back(pin2vertex.size());
  }

  NetList netlist;
  netlist.set_vertex_property(std::move(type), std::move(lx), std::move(ly), std::move(dx), std::move(dy), std::move(area));

  netlist.set_connectivity(std::move(net_span), std::move(pin2vertex), std::move(pin_x_off), std::move(pin_y_off));
  netlist.sort_to_fit();

  for (auto&& i : netlist.report()) {
    INFO(i);
  }

  return netlist;
}

void MacroPlacer::setDataManager(const std::string& idb_json)
{
  _dm->readFormLefDef(idb_json);
}
}  // namespace imp

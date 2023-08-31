#include "MP.hh"

#include <unordered_map>

#include "Annealer.hh"
#include "DataManager.hh"
#include "Design.hh"
#include "Layout.hh"
#include "Logger.hpp"
#include "NetList.hh"
#include "SA.hh"
#include "SeqPair.hh"
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
  auto netlist = plToNetlist();
  netlist.cellClustering(100);
  for (auto&& var : netlist.report()) {
    INFO(var);
  }
  auto sp = makeRandomSeqPair(netlist._num_moveable);
  auto eval = makeSeqPairEvalFn(netlist);
  std::function<void(SeqPair&)> action = SpAction(netlist._num_moveable);

  SASolve(sp, eval, action, 500, 1.5 * netlist._num_moveable, 0.95, 30000);

  for (size_t i = 0; i < 500; i++) {
    double cost = eval(sp);
    if (i % 100 == 0)
      INFO(cost);
  }
  SASolve(sp, eval, action, 500, 1.5 * netlist._num_moveable, 0.99, 5000);
}

NetList MacroPlacer::plToNetlist()
{
  auto design = _dm->get_design();
  auto layout = _dm->get_layout();
  std::vector<int64_t> lx;
  std::vector<int64_t> ly;
  std::vector<int64_t> dx;
  std::vector<int64_t> dy;
  std::vector<int64_t> area;
  std::vector<NetList::VertexType> type;

  for (Instance* v : design->get_instance_list()) {
    lx.push_back(v->get_coordi().get_x());
    ly.push_back(v->get_coordi().get_y());
    dx.push_back(static_cast<int64_t>(v->get_shape().get_width()));
    dy.push_back(static_cast<int64_t>(v->get_shape().get_height()));
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
  auto core = layout->get_core_shape();
  netlist.set_region(core.get_ll_x(), core.get_ll_y(), core.get_width(), core.get_height());
  netlist.set_vertex_property(std::move(type), std::move(lx), std::move(ly), std::move(dx), std::move(dy), std::move(area));

  netlist.set_connectivity(std::move(net_span), std::move(pin2vertex), std::move(pin_x_off), std::move(pin_y_off));
  netlist.sortToFit();

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

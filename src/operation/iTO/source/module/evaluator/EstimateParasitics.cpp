// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
#include "EstimateParasitics.h"
#include "api/TimingEngine.hh"
#include "api/TimingIDBAdapter.hh"

namespace ito {
EstimateParasitics::EstimateParasitics(DbInterface *dbintreface)
    : _db_interface(dbintreface) {
  _timing_engine = _db_interface->get_timing_engine();
  _db_adapter = _timing_engine->get_db_adapter();
  _dbu = _db_interface->get_dbu();
  Flute::readLUT();
}

EstimateParasitics::EstimateParasitics(TimingEngine *timing_engine, int dbu)
    : _timing_engine(timing_engine) {
  _db_adapter = _timing_engine->get_db_adapter();
  _dbu = dbu;
  Flute::readLUT();
}

/**
 * @brief If parasitics have been evaluated, update the changed net stored in
 * _parasitics_invalid. else update rc tree for all net
 *
 */
void EstimateParasitics::excuteParasiticsEstimate() {
  if (_have_estimated_parasitics) {
    for (Net *net : _parasitics_invalid) {
      DesignObject *driver = net->getDriver();
      if (driver) {
        if (_timing_engine->get_ista()->getRcNet(net)) {
          _timing_engine->resetRcTree(net);
        }
        excuteWireParasitic(driver, net, _db_adapter);
      }
    }
    _parasitics_invalid.clear();
  } else {
    estimateAllNetParasitics();
  }
}

/**
 * @brief update rc tree for all net
 *
 */
void EstimateParasitics::estimateAllNetParasitics() {
  LOG_INFO << "estimate all net parasitics start";
  Netlist *design_nl = _timing_engine->get_netlist();
  Net *    net;
  FOREACH_NET(design_nl, net) {
    estimateNetParasitics(net);
  }
  _have_estimated_parasitics = true;
  _parasitics_invalid.clear();
  LOG_INFO << "estimate all net parasitics end";
}

/**
 * @brief update rc for special net
 *
 * @param net
 */
void EstimateParasitics::estimateNetParasitics(Net *net) {
  if (_timing_engine->get_ista()->getRcNet(net)) {
    _timing_engine->resetRcTree(net);
  }
  DesignObject *drvr_pin_port = net->getDriver();
  excuteWireParasitic(drvr_pin_port, net, _db_adapter);
}

/**
 * @brief Re-estimate net with invalid RC values
 *
 * @param drvr_pin_port
 * @param net
 */
void EstimateParasitics::estimateInvalidNetParasitics(DesignObject *drvr_pin_port,
                                                      Net *         net) {

  if (_parasitics_invalid.find(net) != _parasitics_invalid.end() && net) {
    if (_timing_engine->get_ista()->getRcNet(net)) {
      _timing_engine->resetRcTree(net);
    }
    excuteWireParasitic(drvr_pin_port, net, _db_adapter);

    _parasitics_invalid.erase(net);
  }
}

void EstimateParasitics::excuteWireParasitic(DesignObject *drvr_pin_port, Net *curr_net,
                                             TimingDBAdapter *db_adapter) {
  RoutingTree *tree = makeRoutingTree(curr_net, db_adapter, RoutingType::kSteiner);

  if (tree) {
    vector<int> segment_idx;
    vector<int> length_wire;
    tree->drvrToLoadLength(tree->get_root(), segment_idx, length_wire, 0);

    std::vector<std::pair<int, int>> wire_segment_idx;
    std::vector<int>                 length_per_wire;

    tree->segmentIndexAndLength(tree->get_root(), wire_segment_idx, length_per_wire);

    int numb = wire_segment_idx.size();
    for (int i = 0; i != numb; ++i) {
      int      index1 = wire_segment_idx[i].first;
      int      index2 = wire_segment_idx[i].second;
      RctNode *n1 = _timing_engine->makeOrFindRCTreeNode(curr_net, index1);
      RctNode *n2 = _timing_engine->makeOrFindRCTreeNode(curr_net, index2);

      int length_dbu = length_per_wire[i];
      if (length_dbu == 0) {
        _timing_engine->makeResistor(curr_net, n1, n2, 1.0e-3);
      } else {
        std::optional<double> width = std::nullopt;
        double                cap = dynamic_cast<TimingIDBAdapter *>(db_adapter)
                         ->getCapacitance(1, (double)length_dbu / _dbu, width);
        double res = dynamic_cast<TimingIDBAdapter *>(db_adapter)
                         ->getResistance(1, (double)length_dbu / _dbu, width);

        if (curr_net->isClockNet()) {
          cap /= 10.0;
          res /= 10.0;
        // } else {
        //   cap /= 2.0;
        //   res /= 2.0;
        }

        _timing_engine->incrCap(n1, cap / 2.0, true);
        _timing_engine->makeResistor(curr_net, n1, n2, res);
        _timing_engine->incrCap(n2, cap / 2.0, true);
      }
      RctNodeConnectPin(curr_net, index1, n1, tree);
      RctNodeConnectPin(curr_net, index2, n2, tree);
    }

    _timing_engine->updateRCTreeInfo(curr_net);

    delete tree;
  }
}

void EstimateParasitics::RctNodeConnectPin(Net *net, int index, RctNode *rcnode,
                                           RoutingTree *tree) {
  int num_pins = tree->get_pins().size();
  if (tree->get_pin_visit(index) == 1) {
    return;
  }
  if (index < num_pins) {
    tree->set_pin_visit(index);
    RctNode *pin_node = _timing_engine->makeOrFindRCTreeNode(tree->get_pin(index));
    if (index == tree->get_root()->get_id()) {
      _timing_engine->makeResistor(net, pin_node, rcnode, 1.0e-3);
    } else {
      _timing_engine->makeResistor(net, rcnode, pin_node, 1.0e-3);
    }
  }
}

void EstimateParasitics::parasiticsInvalid(Net *net) {
  // printf("EstimateParasitics | parasitics invalid {%s}\n", net->get_name());
  _parasitics_invalid.insert(net);
}

} // namespace ito
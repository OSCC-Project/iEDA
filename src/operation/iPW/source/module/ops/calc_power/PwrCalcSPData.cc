/**
 * @file PwrCalcSPData.cc
 * @author shaozheqing (707005020@qq.com)
 * @brief clac sp data.
 * @version 0.1
 * @date 2023-04-18
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "PwrCalcSPData.hh"

namespace ipower {
/**
 * @brief Get SP data of the port.
 *
 * @param port_name
 * @param inst_cell
 * @return double
 */
double PwrCalcSPData::getSPData(std::string_view port_name, Instance* inst) {
  // find pin
  Pin* pin;
  Pin* the_find_pin = nullptr;
  FOREACH_INSTANCE_PIN(inst, pin) {
    std::string_view pin_name = pin->get_name();
    if (pin_name == port_name) {
      the_find_pin = pin;
      break;
    }
  }
  LOG_FATAL_IF(!the_find_pin) << "inst " << inst->getFullName() << " cell "
                              << inst->get_inst_cell()->get_cell_name()
                              << " not found pin " << port_name;

  // find power vertex
  PwrVertex* the_pwr_vertex = nullptr;
  auto* the_pwr_graph = get_the_pwr_graph();
  auto* the_sta_graph = the_pwr_graph->get_sta_graph();
  auto the_sta_vertex = the_sta_graph->findVertex(the_find_pin);
  if (the_sta_vertex) {
    the_pwr_vertex = the_pwr_graph->staToPwrVertex(*the_sta_vertex);
  } else {
    LOG_FATAL << "not found sta vertex.";
  }

  double sp_value = the_pwr_vertex->getSPData(std::nullopt);

  return sp_value;
}

/**
 * @brief Calc sp data, include analysis conditions.
 *
 * @param expr
 * @param inst
 * @return double
 */
double PwrCalcSPData::calcSPData(LibertyExpr* expr, Instance* inst) {
  double sp_data;
  const char* port_name = expr->get_port();
  auto* left_expr = expr->get_left();
  auto* right_expr = expr->get_right();

  switch (expr->get_op()) {
    /*Monocular calculation.*/
    case LibertyExpr::Operator::kBuffer: {
      sp_data = getSPData(port_name, inst);
      break;
    }

    case LibertyExpr::Operator::kOne: {
      sp_data = 1.0;
      break;
    }

    case LibertyExpr::Operator::kZero: {
      sp_data = 0.0;
      break;
    }

    case LibertyExpr::Operator::kNot: {
      auto left_port_sp_data = calcSPData(left_expr, inst);
      sp_data = 1 - left_port_sp_data;
      break;
    }

    /*Binocular calculation.*/
    case LibertyExpr::Operator::kOr: {
      auto left_port_sp_data = calcSPData(left_expr, inst);
      auto right_port_sp_data = calcSPData(right_expr, inst);

      double p0 = 1.0 - left_port_sp_data;
      double p1 = 1.0 - right_port_sp_data;

      sp_data = 1 - p0 * p1;
      break;
    }

    case LibertyExpr::Operator::kMult:
    case LibertyExpr::Operator::kAnd: {
      auto left_port_sp_data = calcSPData(left_expr, inst);
      auto right_port_sp_data = calcSPData(right_expr, inst);

      sp_data = left_port_sp_data * right_port_sp_data;
      break;
    }

    case LibertyExpr::Operator::kPlus:
    case LibertyExpr::Operator::kXor: {
      auto left_port_sp_data = calcSPData(left_expr, inst);
      auto right_port_sp_data = calcSPData(right_expr, inst);
      double p0 = left_port_sp_data * (1.0 - right_port_sp_data);
      double p1 = (1 - left_port_sp_data) * right_port_sp_data;

      sp_data = p0 + p1;
      break;
    }
  }
  return sp_data;
}

}  // namespace ipower
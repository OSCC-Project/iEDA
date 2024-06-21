/**
 * @file LibClassifyCell.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief classify the liberty cell according the liberty
 * cell、port、arc、function.
 * @version 0.1
 * @date 2023-12-05
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "LibClassifyCell.hh"

#include <concepts>
#include <functional>
#include <tuple>

namespace ista {

template <typename T>
concept HashType = std::is_integral_v<T> || std::is_same_v<T, std::string>;

/**
 * @brief hash function for string or integer.
 *
 * @tparam T
 * @param value
 * @return std::size_t
 */
template <HashType T>
std::size_t Hash(T value)
{
  return std::hash<T>{}(value);
}

/**
 * @brief combine two value hash.
 *
 * @tparam T1
 * @tparam T2
 * @param value1
 * @param value2
 * @return std::size_t
 */
template <HashType T1, HashType T2>
std::size_t CombineHash(T1 value1, T2 value2)
{
  auto h1 = std::hash<T1>{}(value1);
  auto h2 = std::hash<T2>{}(value2);
  auto combined_hash = h1 ^ h2;

  return combined_hash;
}

/**
 * @brief hash cell port.
 *
 * @param port
 * @return std::size_t
 */
std::size_t LibClassifyCell::hashCellPort(LibPort* port)
{
  std::string port_name = port->get_port_name();
  int port_type = static_cast<int>(port->get_port_type());

  return CombineHash(port_name, port_type);
}

/**
 * @brief hash cell port function.
 *
 * @param expr
 * @return std::size_t
 */
std::size_t LibClassifyCell::hashCellPortFuncExpr(RustLibertyExpr* expr)
{
  if (!expr) {
    return 0;
  }

  switch (expr->op) {
    case RustLibertyExprOp::kBuffer:
      return Hash(std::string(expr->port_name));
    case RustLibertyExprOp::kNot: {
      auto* left_expr = rust_get_expr_left(expr);
      auto result = hashCellPortFuncExpr(left_expr);
      rust_free_expr(left_expr);
      return result;
    }

    default: {
      auto* left_expr = rust_get_expr_left(expr);
      auto* right_expr = rust_get_expr_right(expr);

      auto result = (hashCellPortFuncExpr(left_expr) ^ hashCellPortFuncExpr(right_expr)) ^ Hash(static_cast<unsigned>(expr->op));

      rust_free_expr(left_expr);
      rust_free_expr(right_expr);

      return result;
    }
  }
}

/**
 * @brief calculate cell hash.
 *
 * @param the_cell
 * @return std::size_t
 */
std::size_t LibClassifyCell::calculateCellHash(LibCell* the_cell)
{
  std::size_t hash = 0;

  LibPort* port;
  FOREACH_CELL_PORT(the_cell, port)
  {
    hash += hashCellPort(port);
    hash += hashCellPortFuncExpr(port->get_func_expr());
  }
  return hash;
}

/**
 * @brief compare liberty port.
 *
 * @param port1
 * @param port2
 * @return true
 * @return false
 */
bool LibClassifyCell::comparePort(LibPort* port1, LibPort* port2)
{
  return (port1 == nullptr && port2 == nullptr)
         || (port1 != nullptr && port2 != nullptr && Str::equal(port1->get_port_name(), port2->get_port_name())
             && port1->get_port_type() == port2->get_port_type());
}

/**
 * @brief compare liberty func expr.
 *
 * @param expr1
 * @param expr2
 * @return true
 * @return false
 */
bool LibClassifyCell::comparePortFunc(RustLibertyExpr* expr1, RustLibertyExpr* expr2)
{
  if (expr1 == nullptr && expr2 == nullptr) {
    return true;
  }

  if (expr1 != nullptr && expr2 != nullptr && expr1->op == expr2->op) {
    switch (expr1->op) {
      case RustLibertyExprOp::kBuffer:
        return Str::equal(expr1->port_name, expr2->port_name);
      case RustLibertyExprOp::kNot: {
        auto* left_expr1 = rust_get_expr_left(expr1);
        auto* left_expr2 = rust_get_expr_left(expr2);
        bool result = comparePortFunc(left_expr1, left_expr2);
        rust_free_expr(left_expr1);
        rust_free_expr(left_expr2);
        return result;
      }

      default: {
        {
          auto* left_expr1 = rust_get_expr_left(expr1);
          auto* left_expr2 = rust_get_expr_left(expr2);
          bool result = comparePortFunc(left_expr1, left_expr2);
          rust_free_expr(left_expr1);
          rust_free_expr(left_expr2);
          if (!result) {
            return result;
          }
        }
        {
          auto* right_expr1 = rust_get_expr_right(expr1);
          auto* right_expr2 = rust_get_expr_right(expr2);

          bool result = comparePortFunc(right_expr1, right_expr2);
          rust_free_expr(right_expr1);
          rust_free_expr(right_expr2);

          return result;
        }
      }
    }
  }

  return false;
}

/**
 * @brief compare cell port and func.
 *
 * @param cell1
 * @param cell2
 * @return true
 * @return false
 */
bool LibClassifyCell::comparePorts(LibCell* cell1, LibCell* cell2)
{
  bool ret_value = true;
  if (cell1->get_num_port() != cell2->get_num_port()) {
    ret_value = false;
  } else {
    LibPort* port1;
    FOREACH_CELL_PORT(cell1, port1)
    {
      const char* name = port1->get_port_name();
      LibPort* port2 = cell2->get_cell_port_or_port_bus(name);
      if (!(port2 && comparePort(port1, port2) && comparePortFunc(port1->get_func_expr(), port2->get_func_expr()))) {
        ret_value = false;
      }
    }
  }
  return ret_value;
}

/**
 * @brief compare cell timing arc.
 *
 * @param set1
 * @param set2
 * @return true
 * @return false
 */
bool LibClassifyCell::compareTimingArc(LibArcSet* set1, LibArcSet* set2)
{
  return Str::equal(set1->front()->get_src_port(), set2->front()->get_src_port())
         && Str::equal(set1->front()->get_snk_port(), set2->front()->get_snk_port())
         && set1->front()->get_timing_type() == set2->front()->get_timing_type();
}

/**
 * @brief compare cell timing arc sets.
 *
 * @param cell1
 * @param cell2
 * @return true
 * @return false
 */
bool LibClassifyCell::compareTimingArcSets(LibCell* cell1, LibCell* cell2)
{
  bool ret_value = true;
  if (cell1->getCellArcSetCount() != cell2->getCellArcSetCount()) {
    ret_value = false;
  } else {
    LibArcSet* set1;
    FOREACH_CELL_TIMING_ARC_SET(cell1, set1)
    {
      auto set2 = cell2->findLibertyArcSet(set1->front()->get_src_port(), set1->front()->get_snk_port(), set1->front()->get_timing_type());
      if (!(set2 && compareTimingArc(set1, *set2))) {
        ret_value = false;
      }
    }
  }
  return ret_value;
}

/**
 * @brief compare two cell function is the same.
 *
 * @param the_cell1
 * @param the_cell2
 * @return true
 * @return false
 */
bool LibClassifyCell::compareFunction(LibCell* the_cell1, LibCell* the_cell2)
{
  return comparePorts(the_cell1, the_cell2) && compareTimingArcSets(the_cell1, the_cell2);
}

/**
 * @brief classify one lib.
 *
 * @param the_lib
 * @param hash_to_cells
 */
void LibClassifyCell::classifyOneLibCell(LibLibrary* the_lib, std::unordered_map<u_int64_t, Vector<LibCell*>>& hash_to_cells)
{
  LibCell* cell;
  FOREACH_LIB_CELL(the_lib, cell)
  {
    if (cell->isDontUse()) {
      continue;
    }

    unsigned hash = calculateCellHash(cell);
    auto& the_hash_cells = hash_to_cells[hash];
    for (auto* the_cell : the_hash_cells) {
      if (compareFunction(the_cell, cell)) {
        auto& the_cell_class = _func_same_cells[the_cell];
        the_cell_class.push_back(cell);

        auto& cell_class = _func_same_cells[cell];
        cell_class.push_back(the_cell);
      }
    }
    the_hash_cells.push_back(cell);
  }
}

/**
 * @brief classify the cells of libs.
 *
 * @param the_libs
 */
void LibClassifyCell::classifyLibCell(std::vector<LibLibrary*>& the_libs)
{
  std::unordered_map<std::size_t, Vector<LibCell*>> hash_to_cells;
  for (auto* the_lib : the_libs) {
    classifyOneLibCell(the_lib, hash_to_cells);
  }
}

}  // namespace ista
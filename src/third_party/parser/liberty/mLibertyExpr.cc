// OpenSTA, Static Timing Analyzer
// Copyright (c) 2021, Parallax Software, Inc.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
#include "mLibertyExpr.hh"

#include <cstring>

namespace ista {

LibertyExpr::LibertyExpr(Operator op) : _op(op)
{
}

LibertyExprBuilder::LibertyExprBuilder(LibertyPort* expr_port, const char* expr_str) : _expr_port(expr_port), _expr_str(expr_str)
{
}

/**
 * @brief copy the origin str.
 *
 * @param str
 * @return char*
 */
char* LibertyExprBuilder::stringCopy(const char* str)
{
  if (str) {
    char* copy = new char[strlen(str) + 1];
    strcpy(copy, str);
    return copy;
  } else
    return nullptr;
}

/**
 * @brief provide the expr parser string.
 *
 * @param buf
 * @param max_size
 * @return std::size_t
 */
std::size_t LibertyExprBuilder::input(char* buf, size_t max_size)
{
  strncpy(buf, _expr_str.c_str(), max_size);
  int count = strlen(buf);
  _expr_str += count;
  return count;
}

/**
 * @brief make buffer expr.
 *
 * @param port_name
 * @return LibertyExpr*
 */
LibertyExpr* LibertyExprBuilder::makeBufferExpr(const char* port_name)
{
  auto* buffer_expr = new LibertyExpr(LibertyExpr::Operator::kBuffer);
  buffer_expr->set_port(port_name);
  return buffer_expr;
}

/**
 * @brief make not expr.
 *
 * @param expr
 * @return LibertyExpr*
 */
LibertyExpr* LibertyExprBuilder::makeNotExpr(LibertyExpr* expr)
{
  auto* not_expr = new LibertyExpr(LibertyExpr::Operator::kNot);
  not_expr->set_left(expr);

  return not_expr;
}

/**
 * @brief make plus expr.
 *
 * @param expr
 * @return LibertyExpr*
 */
LibertyExpr* LibertyExprBuilder::makePlusExpr(LibertyExpr* left_expr, LibertyExpr* right_expr)
{
  auto* plus_expr = new LibertyExpr(LibertyExpr::Operator::kPlus);
  plus_expr->set_left(left_expr);
  plus_expr->set_right(right_expr);

  return plus_expr;
}

/**
 * @brief make or expr.
 *
 * @param expr
 * @return LibertyExpr*
 */
LibertyExpr* LibertyExprBuilder::makeOrExpr(LibertyExpr* left_expr, LibertyExpr* right_expr)
{
  auto* or_expr = new LibertyExpr(LibertyExpr::Operator::kOr);
  or_expr->set_left(left_expr);
  or_expr->set_right(right_expr);

  return or_expr;
}

/**
 * @brief make mult expr.
 *
 * @param expr
 * @return LibertyExpr*
 */
LibertyExpr* LibertyExprBuilder::makeMultExpr(LibertyExpr* left_expr, LibertyExpr* right_expr)
{
  auto* mult_expr = new LibertyExpr(LibertyExpr::Operator::kMult);
  mult_expr->set_left(left_expr);
  mult_expr->set_right(right_expr);

  return mult_expr;
}

/**
 * @brief make and expr.
 *
 * @param expr
 * @return LibertyExpr*
 */
LibertyExpr* LibertyExprBuilder::makeAndExpr(LibertyExpr* left_expr, LibertyExpr* right_expr)
{
  auto* and_expr = new LibertyExpr(LibertyExpr::Operator::kAnd);
  and_expr->set_left(left_expr);
  and_expr->set_right(right_expr);

  return and_expr;
}

/**
 * @brief make and expr.
 *
 * @param expr
 * @return LibertyExpr*
 */
LibertyExpr* LibertyExprBuilder::makeXorExpr(LibertyExpr* left_expr, LibertyExpr* right_expr)
{
  auto* xor_expr = new LibertyExpr(LibertyExpr::Operator::kXor);
  xor_expr->set_left(left_expr);
  xor_expr->set_right(right_expr);

  return xor_expr;
}

/**
 * @brief execute the expr parser.
 *
 */
void LibertyExprBuilder::execute()
{
  parseBegin();
  parse();
  parseEnd();
}

}  // namespace ista

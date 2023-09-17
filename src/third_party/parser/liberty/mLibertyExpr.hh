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

#pragma once

#include <memory>
#include <string>

namespace ista {

class LibertyPort;

/**
 * @brief The liberty expression.
 *
 */
class LibertyExpr
{
 public:
  enum class Operator
  {
    kBuffer = 1,
    kNot = 2,
    kOr = 3,
    kAnd = 4,
    kXor = 5,
    kOne = 6,
    kZero = 7,
    kPlus = 8,
    kMult = 9

  };
  explicit LibertyExpr(Operator op);
  ~LibertyExpr() = default;

  auto get_op() { return _op; }
  void set_left(LibertyExpr* left) { _left.reset(left); }
  auto* get_left() { return _left.get(); }

  void set_right(LibertyExpr* right) { _right.reset(right); }
  auto* get_right() { return _right.get(); }

  void set_port(const char* port) { _port = port; }
  const char* get_port() { return _port.c_str(); }

  bool isOne() { return _op == Operator::kOne; }
  bool isZero() { return _op == Operator::kZero; }

 private:
  Operator _op;
  std::unique_ptr<LibertyExpr> _left;
  std::unique_ptr<LibertyExpr> _right;
  std::string _port;
};

/**
 * @brief The liberty expression func
 *
 */
class LibertyExprBuilder
{
 public:
  explicit LibertyExprBuilder(LibertyPort* expr_port, const char* expr_str);
  ~LibertyExprBuilder() = default;

  void parseBegin();
  int parse();
  void parseEnd();

  unsigned readLib();

  const char* get_file_name() { return _file_name.c_str(); }
  void incrLineNo() { _line_no++; }
  [[nodiscard]] int get_line_no() const { return _line_no; }

  auto& get_expr_str() { return _expr_str; }

  void clearRecordStr() { _string_buf.erase(); }
  const char* get_record_str() { return _string_buf.c_str(); }
  void recordStr(const char* str) { _string_buf += str; }

  char* stringCopy(const char* str);
  void stringDelete(const char* str) { delete[] str; }

  std::size_t input(char* buf, size_t max_size);
  LibertyExpr* makeBufferExpr(const char* port_name);
  LibertyExpr* makeNotExpr(LibertyExpr* expr);
  LibertyExpr* makePlusExpr(LibertyExpr* left_expr, LibertyExpr* right_expr);
  LibertyExpr* makeOrExpr(LibertyExpr* left_expr, LibertyExpr* right_expr);
  LibertyExpr* makeMultExpr(LibertyExpr* left_expr, LibertyExpr* right_expr);
  LibertyExpr* makeAndExpr(LibertyExpr* left_expr, LibertyExpr* right_expr);
  LibertyExpr* makeXorExpr(LibertyExpr* left_expr, LibertyExpr* right_expr);
  LibertyExpr* makeOneExpr() { return new LibertyExpr(LibertyExpr::Operator::kOne); }
  LibertyExpr* makeZeroExpr() { return new LibertyExpr(LibertyExpr::Operator::kZero); }

  void set_result_expr(LibertyExpr* result_expr) { _result_expr = result_expr; }
  LibertyExpr* get_result_expr() { return _result_expr; }

  void execute();

 private:
  LibertyPort* _expr_port;    //!< The port owned the func attribute.
  std::string _expr_str;      //!< The func expr string.
  LibertyExpr* _result_expr;  //!< The result expr.

  std::string _file_name;    //!< The verilog file name.
  int _line_no = 1;          //!< The verilog file line no.
  std::string _string_buf;   //!< For flex record inner string.
  void* _scanner = nullptr;  //!< The flex scanner.
};

}  // namespace ista

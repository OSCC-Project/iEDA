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
/**
 * @file VerilogReader.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The verilog reader is used for build the netlist database.
 * @version 0.1
 * @date 2021-07-20
 */
#pragma once

#include <cstdlib>
#include <map>

#include "Vector.hh"
#include "include/Config.hh"
#include "log/Log.hh"
#include "string/Str.hh"

namespace ista {

class VerilogReader;
class VerilogModule;
using ieda::Str;

/**
 * @brief The base class of verilog id.
 *
 */
class VerilogID
{
 public:
  explicit VerilogID(const char* id);
  virtual ~VerilogID() = default;

  VerilogID(const VerilogID& orig) = default;
  VerilogID& operator=(const VerilogID& orig) = default;

  virtual VerilogID* copy() { return new VerilogID(*this); }

  virtual unsigned isBusIndexID() { return 0; }
  virtual unsigned isBusSliceID() { return 0; }

  const char* getBaseName() { return _id.c_str(); }
  void setBaseName(std::string&& name) { _id = std::move(name); }
  virtual const char* getName() { return _id.c_str(); }

 protected:
  std::string _id;
};

/**
 * @brief The verilog index id such as gpio[0].
 *
 */
class VerilogIndexID : public VerilogID
{
 public:
  VerilogIndexID(const char* id, int index);
  ~VerilogIndexID() override = default;

  VerilogIndexID(const VerilogIndexID& orig) = default;
  VerilogIndexID& operator=(const VerilogIndexID& orig) = default;

  VerilogID* copy() override { return new VerilogIndexID(*this); }

  unsigned isBusIndexID() override { return 1; }

  [[nodiscard]] int get_index() const { return _index; }

  const char* getName() override { return Str::printf("%s[%d]", _id.c_str(), _index); }

 private:
  int _index;
};

/**
 * @brief The verilog slice id such as gpio[1:0].
 *
 */
class VerilogSliceID : public VerilogID
{
 public:
  VerilogSliceID(const char* id, int range_from, int range_to);
  ~VerilogSliceID() override = default;

  VerilogSliceID(const VerilogSliceID& orig) = default;
  VerilogSliceID& operator=(const VerilogSliceID& orig) = default;

  VerilogID* copy() override { return new VerilogSliceID(*this); }

  unsigned isBusSliceID() override { return 1; }

  [[nodiscard]] int get_range_from() const { return _range_from; }
  void set_range_from(int range_from) { _range_from = range_from; }
  [[nodiscard]] int get_range_to() const { return _range_to; }
  void set_range_to(int range_to) { _range_to = range_to; }

  [[nodiscard]] int get_range_base() { return std::min(_range_from, _range_to); }
  [[nodiscard]] int get_range_max() { return std::max(_range_from, _range_to); }

  const char* getName() override { return Str::printf("%s[%d:%d]", _id.c_str(), _range_from, _range_to); }

  const char* getName(unsigned index) { return Str::printf("%s[%d]", _id.c_str(), index); }

 private:
  int _range_from;
  int _range_to;
};

/**
 * @brief The verilog net expression base class, verilog net expression include
 * constant, id, concat expr way.
 *
 */
class VerilogNetExpr
{
 public:
  explicit VerilogNetExpr(unsigned line_no) : _line_no(line_no) {}
  virtual ~VerilogNetExpr() = default;

  VerilogNetExpr(const VerilogNetExpr& orig) = default;
  VerilogNetExpr& operator=(const VerilogNetExpr& orig) = default;

  virtual VerilogNetExpr* copy() = 0;

  virtual unsigned isIDExpr() { return 0; }
  virtual unsigned isConcatExpr() { return 0; }
  virtual unsigned isConstant() { return 0; }

  [[nodiscard]] unsigned get_line_no() const { return _line_no; }

  virtual VerilogID* get_verilog_id()
  {
    LOG_FATAL << "not implement.";
    return nullptr;
  }

  virtual void set_verilog_id(std::unique_ptr<VerilogID> verilog_id) { LOG_FATAL << "not implement."; }

 private:
  unsigned _line_no = 0;
};

/**
 * @brief The verilog id expression way.
 *
 */
class VerilogNetIDExpr : public VerilogNetExpr
{
 public:
  explicit VerilogNetIDExpr(VerilogID* verilog_id, unsigned line_no);
  ~VerilogNetIDExpr() override = default;

  VerilogNetIDExpr(const VerilogNetIDExpr& orig);
  VerilogNetIDExpr& operator=(const VerilogNetIDExpr& orig);

  VerilogNetExpr* copy() override { return new VerilogNetIDExpr(*this); }

  [[nodiscard]] unsigned isIDExpr() override { return 1; }
  VerilogID* get_verilog_id() override { return _verilog_id.get(); }

  void set_verilog_id(std::unique_ptr<VerilogID> verilog_id) override { _verilog_id = std::move(verilog_id); }

 private:
  std::unique_ptr<VerilogID> _verilog_id;
};

/**
 * @brief The verilog concatenation expression way.
 *
 */
class VerilogNetConcatExpr : public VerilogNetExpr
{
 public:
  explicit VerilogNetConcatExpr(Vector<std::unique_ptr<VerilogNetExpr>>&& verilog_id_concat, unsigned line_no);
  ~VerilogNetConcatExpr() override = default;

  VerilogNetConcatExpr(const VerilogNetConcatExpr& orig);
  VerilogNetConcatExpr& operator=(const VerilogNetConcatExpr& orig);

  VerilogNetExpr* copy() override { return new VerilogNetConcatExpr(*this); }

  [[nodiscard]] unsigned isConcatExpr() override { return 1; }
  auto& get_verilog_id_concat() { return _verilog_id_concat; }

  VerilogNetExpr* getVerilogIdExpr(unsigned index)
  {
    LOG_FATAL_IF(index >= _verilog_id_concat.size());
    // for verilog expr, the max bit first, so we need reverse the index.
    return _verilog_id_concat[_verilog_id_concat.size() - index - 1].get();
  }

  void setVerilogIdExpr(unsigned index, VerilogNetExpr* new_net_expr)
  {
    LOG_FATAL_IF(index >= _verilog_id_concat.size());
    _verilog_id_concat[_verilog_id_concat.size() - index - 1].reset(new_net_expr);
  }

 private:
  Vector<std::unique_ptr<VerilogNetExpr>> _verilog_id_concat;  //!< such as { 2'b00, _0_ }
};

/**
 * @brief The verilog constant expression, such as 1'b0, 1'b1.
 *
 */
class VerilogConstantExpr : public VerilogNetExpr
{
 public:
  VerilogConstantExpr(const char* constant, unsigned line_no);
  ~VerilogConstantExpr() override = default;

  VerilogConstantExpr(const VerilogConstantExpr& orig);
  VerilogConstantExpr& operator=(const VerilogConstantExpr& orig);

  VerilogNetExpr* copy() override { return new VerilogConstantExpr(*this); }

  [[nodiscard]] unsigned isConstant() override { return 1; }
  VerilogID* get_verilog_id() override { return _verilog_id.get(); }

 private:
  std::unique_ptr<VerilogID> _verilog_id;  //!< 1'b0 or 1'b1.
};

/**
 * @brief The base class for verilog stmt,include module dcl, module instance,
 * module assign.
 *
 */
class VerilogStmt
{
 public:
  explicit VerilogStmt(int line);
  virtual ~VerilogStmt() = default;

  VerilogStmt(const VerilogStmt& orig) = default;
  VerilogStmt& operator=(const VerilogStmt& orig) = default;

  virtual VerilogStmt* copy() { return new VerilogStmt(*this); }

  [[nodiscard]] int get_line() const { return _line; }

  virtual unsigned isModuleInstStmt() { return 0; }
  virtual unsigned isModuleAssignStmt() { return 0; }
  virtual unsigned isVerilogDclStmt() { return 0; }
  virtual unsigned isVerilogDclsStmt() { return 0; }
  virtual unsigned isModuleStmt() { return 0; }

 private:
  int _line;
};

/**
 * @brief The verilog module port connection of module instance statement.
 *
 */
class VerilogPortConnect
{
 public:
  VerilogPortConnect() = default;
  virtual ~VerilogPortConnect() = default;

  VerilogPortConnect(const VerilogPortConnect& orig) = default;
  VerilogPortConnect& operator=(const VerilogPortConnect& orig) = default;

  virtual VerilogPortConnect* copy() = 0;
};

/**
 * @brief The port connection such as .port_id(net_id).
 *
 */
class VerilogPortRefPortConnect : public VerilogPortConnect
{
 public:
  VerilogPortRefPortConnect(VerilogID* port_id, VerilogNetExpr* net_id);
  ~VerilogPortRefPortConnect() override = default;

  VerilogPortRefPortConnect(const VerilogPortRefPortConnect& orig);
  VerilogPortRefPortConnect& operator=(const VerilogPortRefPortConnect& orig);

  VerilogPortConnect* copy() override { return new VerilogPortRefPortConnect(*this); }

  VerilogID* get_port_id() { return _port_id.get(); }
  VerilogNetExpr* get_net_expr() { return _net_expr.get(); }
  void set_net_expr(std::unique_ptr<VerilogNetExpr>&& net_expr) { _net_expr = std::move(net_expr); }
  std::unique_ptr<VerilogNetExpr>& takeNetExpr() { return _net_expr; }

 private:
  std::unique_ptr<VerilogID> _port_id;
  std::unique_ptr<VerilogNetExpr> _net_expr;
};

/**
 * @brief Verilog instance stmt.
 *
 */
class VerilogInst : public VerilogStmt
{
 public:
  VerilogInst(const char* liberty_cell_name, const char* inst_name,
              std::vector<std::unique_ptr<VerilogPortRefPortConnect>>&& port_connection, int line);
  ~VerilogInst() override = default;

  VerilogInst(const VerilogInst& orig);
  VerilogInst& operator=(const VerilogInst& orig);

  VerilogStmt* copy() override { return new VerilogInst(*this); }

  unsigned isModuleInstStmt() override { return 1; }

  const char* get_inst_name() { return _inst_name.c_str(); }
  void set_inst_name(std::string&& inst_name) { _inst_name = std::move(inst_name); }

  const char* get_cell_name() { return _cell_name.c_str(); }
  auto& get_port_connections() { return _port_connections; }
  std::unique_ptr<VerilogNetExpr> getPortConnectNet(VerilogModule* parent_module, VerilogModule* inst_module, VerilogID* port_id,
                                                    std::optional<std::pair<int, int>> bus_size_range);

 private:
  std::string _inst_name;
  std::string _cell_name;

  std::vector<std::unique_ptr<VerilogPortRefPortConnect>> _port_connections;
};

/**
 * @brief Verilog assign stmt.
 *
 */
class VerilogAssign : public VerilogStmt
{
 public:
  VerilogAssign(VerilogNetExpr* left_net_expr, VerilogNetExpr* right_net_expr, int line);
  ~VerilogAssign() override = default;
  unsigned isModuleAssignStmt() { return 1; }

  VerilogNetExpr* get_left_net_expr() { return _left_net_expr.get(); }
  VerilogNetExpr* get_right_net_expr() { return _right_net_expr.get(); }

 private:
  std::unique_ptr<VerilogNetExpr> _left_net_expr;
  std::unique_ptr<VerilogNetExpr> _right_net_expr;
};

/**
 * @brief iterate the inst port connect.
 *
 */
#define FOREACH_VERILOG_PORT_CONNECT(inst, port_connect) for (auto& port_connect : inst->get_port_connections())

/**
 * @brief The wire or port declaration.
 *
 */
class VerilogDcl : public VerilogStmt
{
 public:
  enum class DclType : int
  {
    kInput = 0,
    kInout = 1,
    kOutput = 2,
    kSupply0 = 3,
    kSupply1 = 4,
    kTri = 5,
    kWand = 6,
    kWire = 7,
    kWor = 7
  };
  VerilogDcl(DclType dcl_type, const char* dcl_name, int line);
  ~VerilogDcl() override = default;

  VerilogDcl(const VerilogDcl& orig) = default;
  VerilogDcl& operator=(const VerilogDcl& orig) = default;

  VerilogStmt* copy() override { return new VerilogDcl(*this); }

  unsigned isVerilogDclStmt() override { return 1; }

  DclType get_dcl_type() { return _dcl_type; }
  const char* get_dcl_name() { return _dcl_name.c_str(); }
  void set_dcl_name(std::string&& dcl_name) { _dcl_name = std::move(dcl_name); }

  void set_range(std::pair<int, int> range) { _range = range; }
  auto& get_range() { return _range; }

 private:
  DclType _dcl_type;
  std::string _dcl_name;
  std::optional<std::pair<int, int>> _range;
};

/**
 * @brief The mutiple verilg dcl.
 *
 */
class VerilogDcls : public VerilogStmt
{
 public:
  VerilogDcls(std::vector<std::unique_ptr<VerilogDcl>>&& verilog_dcls, int line);
  ~VerilogDcls() override = default;

  VerilogDcls(const VerilogDcls& orig);
  VerilogDcls& operator=(const VerilogDcls& orig);

  VerilogStmt* copy() override { return new VerilogDcls(*this); }

  unsigned isVerilogDclsStmt() override { return 1; }

  auto& get_verilog_dcls() { return _verilog_dcls; }
  auto get_dcl_num() { return _verilog_dcls.size(); }

 private:
  std::vector<std::unique_ptr<VerilogDcl>> _verilog_dcls;
};

/**
 * @brief The verilog module class.
 *
 */
class VerilogModule : public VerilogStmt
{
 public:
  enum class PortDclType : int
  {
    kInput = 0,
    kInputWire = 1,
    kInout = 2,
    kInoutReg = 3,
    kInoutWire = 4,
    kOutput = 5,
    kOputputWire = 6,
    kOutputReg = 7
  };

  VerilogModule(const char* module_name, int line);
  ~VerilogModule() override = default;

  unsigned isModuleStmt() override { return 1; }

  const char* get_module_name() { return _module_name.c_str(); }

  void set_module_stmts(std::vector<std::unique_ptr<VerilogStmt>>&& module_stmts) { _module_stmts = std::move(module_stmts); }

  void addStmt(std::unique_ptr<VerilogStmt> module_stmt) { _module_stmts.emplace_back(std::move(module_stmt)); }
  void eraseStmt(VerilogStmt* the_stmt)
  {
    auto ret = std::erase_if(_module_stmts, [the_stmt](auto& stmt) { return stmt.get() == the_stmt; });
    assert(ret != 0);
  }

  auto& get_module_stmts() { return _module_stmts; }

  void set_port_list(std::vector<std::unique_ptr<VerilogID>>&& port_list) { _port_list = std::move(port_list); }

  bool isPort(const char* name)
  {
    auto it = std::find_if(_port_list.begin(), _port_list.end(),
                           [&name](const std::unique_ptr<VerilogID>& id) { return Str::equal(id->getBaseName(), name); });
    return it != _port_list.end();
  }

  VerilogStmt* findDclStmt(const char* name, bool is_need_range = false)
  {
    auto it = std::find_if(_module_stmts.begin(), _module_stmts.end(), [&name, is_need_range](const std::unique_ptr<VerilogStmt>& stmt) {
      if (stmt->isVerilogDclStmt()) {
        if (Str::equal(dynamic_cast<VerilogDcl*>(stmt.get())->get_dcl_name(), name)) {
          return true;
        }
      } else if (stmt->isVerilogDclsStmt()) {
        auto* dcls = dynamic_cast<VerilogDcls*>(stmt.get());

        for (auto& dcl : dcls->get_verilog_dcls()) {
          if (Str::equal(dynamic_cast<VerilogDcl*>(dcl.get())->get_dcl_name(), name)) {
            return true;
          }
        }
      }

      return false;
    });
    if (it != _module_stmts.end()) {
      return it->get();
    }
    return nullptr;
  }

  void flattenModule(VerilogModule* parent_module, VerilogInst* inst_stmt, VerilogReader* verilog_reader);

 private:
  std::string _module_name;
  std::vector<std::unique_ptr<VerilogID>> _port_list;

  std::vector<std::unique_ptr<VerilogStmt>> _module_stmts;
};

/**
 * @brief iterate the verilog stmt.
 *
 */
#define FOR_EACH_VERILOG_STMT(module, stmt) for (auto& stmt : module->get_module_stmts())

/**
 * @brief Verilog reader class.
 *
 */
class VerilogReader
{
 public:
  VerilogReader() = default;
  ~VerilogReader() = default;

  bool read(const char* filename);
#ifdef ZLIB_FOUND
  void parseBegin(gzFile fp);
  void parseEnd(gzFile fp);
#endif
  void parseBegin(FILE* fp);
  int parse();
  void parseEnd(FILE* fp);

  auto& get_file_name() { return _file_name; }
  void incrLineNo() { _line_no++; }
  [[nodiscard]] int get_line_no() const { return _line_no; }

#ifdef ZLIB_FOUND
  void getChars(char* buf, int& result, size_t max_size)
  {
    char* status = gzgets(_verilog_in, buf, max_size);
    if (status == Z_NULL) {
      result = 0;
    } else {
      result = strlen(buf);
    }
  }
#endif

  void clearRecordStr() { _string_buf.erase(); }
  const char* get_record_str() { return _string_buf.c_str(); }
  void recordStr(const char* str) { _string_buf += str; }

  auto& get_verilog_modules() { return _verilog_modules; }
  VerilogModule* findModule(const char* module_name);

  VerilogDcls* makeDcl(VerilogDcl::DclType dcl_type, std::vector<const char*>&& dcl_args, int line);
  VerilogDcls* makeDcl(VerilogDcl::DclType dcl_type, std::vector<const char*>&& dcl_args, int line, std::pair<int, int> range);

  VerilogPortRefPortConnect* makePortConnect(VerilogID* port_id, VerilogNetExpr* net_id);

  VerilogID* makeVerilogID(const char* id);
  VerilogID* makeVerilogID(const char* id, int index);
  VerilogID* makeVerilogID(const char* id, int range_from, int range_to);

  VerilogNetExpr* makeVerilogNetExpr(VerilogID* verilog_id, int line);
  VerilogNetExpr* makeVerilogNetExpr(Vector<std::unique_ptr<VerilogNetExpr>>&& verilog_id_concat, int line);

  VerilogNetExpr* makeVerilogNetExpr(const char* constant, int line);

  VerilogInst* makeModuleInst(const char* liberty_cell_name, const char* inst_name,
                              std::vector<std::unique_ptr<VerilogPortRefPortConnect>>&& port_connection, int line);
  VerilogAssign* makeModuleAssign(VerilogNetExpr* left_net_expr, VerilogNetExpr* right_net_expr, int line);

  VerilogModule* makeModule(const char* module_name, std::vector<std::unique_ptr<VerilogStmt>>&& module_stmts, int line);

  VerilogModule* makeModule(const char* module_name, std::vector<std::unique_ptr<VerilogID>>&& port_list,
                            std::vector<std::unique_ptr<VerilogStmt>>&& module_stmts, int line);

  VerilogModule* flattenModule(const char* module_name);

 private:
  std::vector<std::unique_ptr<VerilogModule>> _verilog_modules;
  std::map<std::string, VerilogModule*> _str2Module;

  std::string _file_name;    //!< The verilog file name.
  int _line_no = 1;          //!< The verilog file line no.
  std::string _string_buf;   //!< For flex record inner string.
  void* _scanner = nullptr;  //!< The flex scanner.
#ifdef ZLIB_FOUND
  gzFile _verilog_in = nullptr;  //!< The verilog file stream.
#endif
};

}  // namespace ista

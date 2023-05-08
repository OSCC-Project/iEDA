/**
 * @file verilog_writer.h
 * @author shy long (longshy@pcl.ac.cn)
 * @brief
 * @version 0.1
 * @date 2021-12-03
 */
#pragma once

#include <stdio.h>

#include <set>
#include <string>
#include <vector>

#include "IdbDesign.h"
#include "IdbEnum.h"
#include "def_service.h"

namespace idb {

class IdbInstance;

class VerilogWriter
{
 public:
  VerilogWriter(const char* file_name, std::set<std::string>& exclude_cell_names, IdbDesign& idb_design);
  ~VerilogWriter();

  void writeModule();
  bool isNeedEscape(const std::string& name);
  std::string escapeName(const std::string& name);

 protected:
  void writePorts();
  void writePortDcls();
  void writeWire();
  void writeInstances();
  void writeInstance(IdbInstance* inst);

 private:
  const char* _file_name;
  std::set<std::string> _exclude_cell_names;

  FILE* _stream;
  IdbDesign& _idb_design;
};
}  // namespace idb

/*
 * @Author: S.J Chen
 * @Date: 2022-02-16 20:03:35
 * @LastEditTime: 2022-12-11 20:15:52
 * @LastEditors: sjchanson 13560469332@163.com
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/wrapper/DBWrapper.hh
 * Contact : https://github.com/sjchanson
 */

#ifndef IPL_DB_WRAPPER_H
#define IPL_DB_WRAPPER_H

#include <string>

#include "data/Design.hh"
#include "data/Layout.hh"

namespace ipl {

class DBWrapper
{
 public:
  DBWrapper()                 = default;
  DBWrapper(const DBWrapper&) = delete;
  DBWrapper(DBWrapper&&)      = delete;
  virtual ~DBWrapper()        = default;

  DBWrapper& operator=(const DBWrapper&) = delete;
  DBWrapper& operator=(DBWrapper&&) = delete;

  // Layout.
  virtual const Layout* get_layout() const = 0;

  // Design.
  virtual Design* get_design() const = 0;

  // Function.
  virtual void writeDef(std::string file_name) = 0;
  virtual void updateFromSourceDataBase()      = 0;
  virtual void updateFromSourceDataBase(std::vector<std::string> inst_list) = 0;
  virtual void writeBackSourceDatabase()       = 0;
  virtual void initInstancesForFragmentedRow() = 0;
  virtual void saveVerilogForDebug(std::string path) = 0;
};

}  // namespace ipl

#endif
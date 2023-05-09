/*
 * @Author: Li Jiangkao
 * @Date: 2022-02-24 16:19:39
 * @LastEditTime: 2022-02-24 17:27:03
 * @LastEditors: Please set LastEditors
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: /home/lixingquan/EDA/PCNL-EDA/LJK-iEDA/iEDA/src/iPL/src/solver/MacroPlacer/wrapper/IPLDBWrapper.hh
 */
#pragma once

#include <map>
#include <string>
#include <vector>

#include "DBWrapper.hh"
#include "IPLDBWDatabase.hh"

namespace ipl::imp {

class IPLDBWrapper : public DBWrapper
{
 public:
  IPLDBWrapper() = delete;
  explicit IPLDBWrapper(ipl::PlacerDB* ipl_db);
  ~IPLDBWrapper() override;

  // Layout
  const FPLayout* get_layout() const { return _iplw_database->_layout; }

  // Design
  FPDesign* get_design() const { return _iplw_database->_design; }

  // Function
  void writeDef(string file_name) override{};
  void writeBackSourceDataBase() override;

 private:
  IPLDBWDatabase* _iplw_database;

  void wrapIPLData();
  void wrapLayout(const ipl::Layout* ipl_layout);
  void wrapDesign(ipl::Design* ipl_design);
  void wrapInstancelist(ipl::Design* ipl_design);
  void wrapNetlist(ipl::Design* ipl_design);
  FPPin* wrapPin(ipl::Pin* ipl_pin);
};
}  // namespace ipl::imp

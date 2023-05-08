/*
 * @Author: your name
 * @Date: 2022-02-24 14:55:44
 * @LastEditTime: 2022-02-24 16:01:04
 * @LastEditors: Please set LastEditors
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: /home/lixingquan/EDA/PCNL-EDA/LJK-iEDA/iEDA/src/iPL/src/solver/MacroPlacer/wrapper/DBWrapper.hh
 */
#pragma once

#include "database/FPDesign.hh"
#include "database/FPLayout.hh"

namespace ipl::imp {

class DBWrapper
{
 public:
  DBWrapper() = default;
  virtual ~DBWrapper() = default;

  // Layout
  virtual const FPLayout* get_layout() const = 0;

  // Design
  virtual FPDesign* get_design() const = 0;

  // Function
  virtual void writeDef(string file_name) = 0;
  virtual void writeBackSourceDataBase() = 0;
};

}  // namespace ipl::imp
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
#include "ScriptEngine.hh"
#include "py_register_config.h"
#include "py_register_eval.h"
#include "py_register_feature.h"
#include "py_register_flow.h"
#include "py_register_icts.h"
#include "py_register_idb.h"
#include "py_register_idrc.h"
#include "py_register_ifp.h"
#include "py_register_ino.h"
#include "py_register_inst.h"
#include "py_register_ipdn.h"
#include "py_register_ipl.h"
#include "py_register_ipw.h"
#include "py_register_irt.h"
#include "py_register_ista.h"
#include "py_register_ito.h"
#include "py_register_report.h"
#include "py_register_vec.h"
#include "python_module.h"

namespace python_interface {

PYBIND11_MODULE(ieda_py, m)
{
  register_config(m);
  register_flow(m);
  register_icts(m);
  register_idb(m);
  register_idb_op(m);
  register_idrc(m);
  register_ifp(m);
  register_ino(m);
  register_inst(m);
  register_ipdn(m);
  register_ipl(m);
  register_irt(m);
  register_ista(m);
  register_ipw(m);
  register_ito(m);
  register_report(m);
  register_feature(m);
  register_eval(m);
  register_vectorization(m);
}

}  // namespace python_interface
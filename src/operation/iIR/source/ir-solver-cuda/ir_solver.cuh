// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of
// Sciences Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan
// PSL v2. You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file ir_solver.cuh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The ir cuda solver.
 * @version 0.1
 * @date 2025-04-19
 *
 */

#pragma once

#include <vector>
#include <Eigen/Sparse>

namespace iir {

/**
 * @brief ir cg solver.
 * 
 * @param A 
 * @param b 
 * @param x0 
 * @param tol 
 * @param max_iter 
 * @return * std::vector<double> 
 */
std::vector<double> ir_cg_solver(Eigen::SparseMatrix<double>& A,
                                 Eigen::VectorXd& b,
                                 Eigen::VectorXd& x0,
                                 const double tol, const int max_iter, double lambda);

}
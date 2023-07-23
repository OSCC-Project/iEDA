/**
 * @file Evaluator.hh
 * @author Fuxing Huang (fxxhuang@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-07-13
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef IMP_EVALUATOR_H
#define IMP_EVALUATOR_H

namespace imp {

template <typename T>
T hpwl(int num_nets, T* x, T* y, int* nets, int* pins, T* x_off, T* y_off, int num_threads = 1);

}  // namespace imp

#endif
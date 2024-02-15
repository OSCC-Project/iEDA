/*******************************************************************************
 * MIT License
 *
 * This file is part of Mt-KaHyPar.
 *
 * Copyright (C) 2019 Tobias Heuer <tobias.heuer@kit.edu>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 ******************************************************************************/

#pragma once

#include <type_traits>

#if defined(MT_KAHYPAR_LIBRARY_MODE) || !defined(KAHYPAR_ENABLE_THREAD_PINNING)
#include "tbb/task_arena.h"
// If we use the C or Python interface or thread pinning is disabled, the cpu ID to
// which the current thread is assigned to is not unique. We therefore use the slot index
// of the current task arena as unique thread ID. Note that the ID can be negative if
// the task scheduler is not initialized.
#define THREAD_ID std::max(0, tbb::this_task_arena::current_thread_index())
#else
#ifdef __linux__
#include <sched.h>
#define THREAD_ID sched_getcpu()
#elif _WIN32
#include <processthreadsapi.h>
#define THREAD_ID GetCurrentProcessorNumber()
#endif
#endif

#include "kahypar-resources/macros.h"

#define SPECIALIZATION(EXPR, TYPE)          \
  template<bool T = EXPR>                   \
  std::enable_if_t<T, TYPE>

#define TRUE_SPECIALIZATION(EXPR, TYPE)     \
  template<bool T = EXPR>                   \
  std::enable_if_t<T, TYPE>

#define FALSE_SPECIALIZATION(EXPR, TYPE)    \
  template<bool T = EXPR>                   \
  std::enable_if_t<!T, TYPE>

#if (defined(__GNUC__) || defined(__clang__)) && defined(NDEBUG)
#define MT_KAHYPAR_ATTRIBUTE_ALWAYS_INLINE __attribute__ ((always_inline)) inline
#else
#define MT_KAHYPAR_ATTRIBUTE_ALWAYS_INLINE
#endif

#define HEAVY_ASSERT0(cond) \
  !(enable_heavy_assert) ? (void)0 : [&]() { ASSERT(cond); } ()
#define HEAVY_ASSERT1(cond, msg) \
  !(enable_heavy_assert) ? (void)0 : [&]() { ASSERT(cond, msg); } ()

#ifdef KAHYPAR_ENABLE_HEAVY_PREPROCESSING_ASSERTIONS
  #define HEAVY_PREPROCESSING_ASSERT_1(cond) ASSERT(cond)
  #define HEAVY_PREPROCESSING_ASSERT_2(cond, msg) ASSERT(cond, msg)
#else
  #define HEAVY_PREPROCESSING_ASSERT_1(cond) HEAVY_ASSERT0(cond)
  #define HEAVY_PREPROCESSING_ASSERT_2(cond, msg) HEAVY_ASSERT1(cond, msg)
#endif

#ifdef KAHYPAR_ENABLE_HEAVY_COARSENING_ASSERTIONS
  #define HEAVY_COARSENING_ASSERT_1(cond) ASSERT(cond)
  #define HEAVY_COARSENING_ASSERT_2(cond, msg) ASSERT(cond, msg)
#else
  #define HEAVY_COARSENING_ASSERT_1(cond) HEAVY_ASSERT0(cond)
  #define HEAVY_COARSENING_ASSERT_2(cond, msg) HEAVY_ASSERT1(cond, msg)
#endif

#ifdef KAHYPAR_ENABLE_HEAVY_INITIAL_PARTITIONING_ASSERTIONS
  #define HEAVY_INITIAL_PARTITIONING_ASSERT_1(cond) ASSERT(cond)
  #define HEAVY_INITIAL_PARTITIONING_ASSERT_2(cond, msg) ASSERT(cond, msg)
#else
  #define HEAVY_INITIAL_PARTITIONING_ASSERT_1(cond) HEAVY_ASSERT0(cond)
  #define HEAVY_INITIAL_PARTITIONING_ASSERT_2(cond, msg) HEAVY_ASSERT1(cond, msg)
#endif

#ifdef KAHYPAR_ENABLE_HEAVY_REFINEMENT_ASSERTIONS
  #define HEAVY_REFINEMENT_ASSERT_1(cond) ASSERT(cond)
  #define HEAVY_REFINEMENT_ASSERT_2(cond, msg) ASSERT(cond, msg)
#else
  #define HEAVY_REFINEMENT_ASSERT_1(cond) HEAVY_ASSERT0(cond)
  #define HEAVY_REFINEMENT_ASSERT_2(cond, msg) HEAVY_ASSERT1(cond, msg)
#endif

#define HEAVY_ASSERT_(TYPE, N) HEAVY_ ## TYPE ## _ASSERT_ ## N
#define HEAVY_ASSERT_EVAL(TYPE, N) HEAVY_ASSERT_(TYPE, N)

// Heavy assertions are assertions which increase the complexity of the scope
// which they are executed in by an polynomial factor. In debug mode you are often only
// interested in certain phase of the multilevel paradigm. However, when enabling all assertions
// it can take a while to reach the point which you are really interested in, because heavy assertions
// radicaly downgrade the performance of the application. Therefore such assertions should be packed
// in a heavy assertion macro. Heavy assertions can be enabled via cmake flag for specific phase or for
// specific scope by adding
// static constexpr bool enable_heavy_assert = false;
// to the corresponding scope.
#define HEAVY_PREPROCESSING_ASSERT(...) EXPAND(HEAVY_ASSERT_EVAL(PREPROCESSING, EXPAND(NARG(__VA_ARGS__)))(__VA_ARGS__))
#define HEAVY_COARSENING_ASSERT(...) EXPAND(HEAVY_ASSERT_EVAL(COARSENING, EXPAND(NARG(__VA_ARGS__)))(__VA_ARGS__))
#define HEAVY_INITIAL_PARTITIONING_ASSERT(...) EXPAND(HEAVY_ASSERT_EVAL(INITIAL_PARTITIONING, EXPAND(NARG(__VA_ARGS__)))(__VA_ARGS__))
#define HEAVY_REFINEMENT_ASSERT(...) EXPAND(HEAVY_ASSERT_EVAL(REFINEMENT, EXPAND(NARG(__VA_ARGS__)))(__VA_ARGS__))

// In windows unisgned long != size_t
#define UL(X) (size_t) X
#ifdef STR
#undef STR
#endif
#define STR(X) std::to_string(X)
#define STREAM2STR(X) static_cast<std::stringstream>(X).str();

// Info, Warning and Error Output Macros
#define GREEN "\033[1;92m"
#define CYAN "\033[1;96m"
#define YELLOW "\033[1;93m"
#define RED "\033[1;91m"
#define WHITE "\033[1;97m"
#define BOLD "\033[1m"
#define END "\033[0m"
#define INFO(msg) LOG << CYAN << "[INFO]" << END << msg
#ifdef WARNING
#undef WARNING
#endif
#define WARNING(msg) LOG << YELLOW << "[WARNING]" << END << msg
#define ERR(msg) LOG << RED << "[ERROR]" << END << msg; std::exit(-1)

#ifdef MT_KAHYPAR_LIBRARY_MODE
#define ALGO_SWITCH(warning_msg, error_msg, context_variable, alternative_value) \
  ERR(error_msg);
#else
#define ALGO_SWITCH(warning_msg, error_msg, context_variable, alternative_value) \
  WARNING(warning_msg);                                                          \
  char answer = 'N';                                                             \
  std::cin >> answer;                                                            \
  answer = std::toupper(answer);                                                 \
  if (answer == 'Y') {                                                           \
    context_variable = alternative_value;                                        \
  } else {                                                                       \
    ERR(error_msg);                                                            \
  }
#endif

#ifdef KAHYPAR_ENABLE_EXPERIMENTAL_FEATURES
#define ENABLE_EXPERIMENTAL_FEATURES(X) X
#else
#define ENABLE_EXPERIMENTAL_FEATURES(X)
#endif

#ifdef KAHYPAR_ENABLE_LARGE_K_PARTITIONING_FEATURES
#define ENABLE_LARGE_K(X) X
#else
#define ENABLE_LARGE_K(X)
#endif

#ifdef KAHYPAR_ENABLE_HIGHEST_QUALITY_FEATURES
#define ENABLE_HIGHEST_QUALITY(X) X
#else
#define ENABLE_HIGHEST_QUALITY(X)
#endif

#ifdef KAHYPAR_ENABLE_GRAPH_PARTITIONING_FEATURES
#define ENABLE_GRAPHS(X) X
#else
#define ENABLE_GRAPHS(X)
#endif

#ifdef KAHYPAR_ENABLE_GRAPH_PARTITIONING_FEATURES
#ifdef KAHYPAR_ENABLE_HIGHEST_QUALITY_FEATURES
#define ENABLE_HIGHEST_QUALITY_FOR_GRAPHS(X) X
#else
#define ENABLE_HIGHEST_QUALITY_FOR_GRAPHS(X)
#endif
#else
#define ENABLE_HIGHEST_QUALITY_FOR_GRAPHS(X)
#endif

#ifdef KAHYPAR_ENABLE_SOED_METRIC
#define ENABLE_SOED(X) X
#else
#define ENABLE_SOED(X)
#endif

#ifdef KAHYPAR_ENABLE_STEINER_TREE_METRIC
#define ENABLE_STEINER_TREE(X) X
#else
#define ENABLE_STEINER_TREE(X)
#endif

#define COMMA ,

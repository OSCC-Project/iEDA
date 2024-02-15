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

#include "gmock/gmock.h"

#include "mt-kahypar/definitions.h"

namespace mt_kahypar::tests {

using Hypergraph = ds::StaticHypergraph;
using PartitionedHypergraph = ds::PartitionedHypergraph<Hypergraph, ds::ConnectivityInfo>;
using HighResClockTimepoint = std::chrono::time_point<std::chrono::high_resolution_clock>;

using HypergraphTestTypes = ::testing::Types<ds::StaticHypergraph
                                             ENABLE_HIGHEST_QUALITY(COMMA ds::DynamicHypergraph)>;
using GraphTestTypes = ::testing::Types<ds::StaticGraph
                                        ENABLE_HIGHEST_QUALITY_FOR_GRAPHS(COMMA ds::DynamicGraph)>;
using GraphAndHypergraphTestTypes = ::testing::Types<ds::StaticHypergraph
                                                     ENABLE_GRAPHS(COMMA ds::StaticGraph)
                                                     ENABLE_HIGHEST_QUALITY(COMMA ds::DynamicHypergraph)
                                                     ENABLE_HIGHEST_QUALITY_FOR_GRAPHS(COMMA ds::DynamicGraph)>;


using HypergraphTestTypeTraits = ::testing::Types<StaticHypergraphTypeTraits
                                                  ENABLE_HIGHEST_QUALITY(COMMA DynamicHypergraphTypeTraits)
                                                  ENABLE_LARGE_K(COMMA LargeKHypergraphTypeTraits)>;
using GraphTestTypeTraits = ::testing::Types<StaticGraphTypeTraits
                                             ENABLE_HIGHEST_QUALITY_FOR_GRAPHS(COMMA DynamicGraphTypeTraits)>;

}  // namespace mt_kahypar::tests

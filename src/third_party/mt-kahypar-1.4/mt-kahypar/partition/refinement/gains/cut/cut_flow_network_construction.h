/*******************************************************************************
 * MIT License
 *
 * This file is part of Mt-KaHyPar.
 *
 * Copyright (C) 2023 Tobias Heuer <tobias.heuer@kit.edu>
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

#include "mt-kahypar/datastructures/hypergraph_common.h"
#include "mt-kahypar/partition/context.h"

namespace mt_kahypar {

/**
 * This struct is used by the flow network construction algorithm
 * to determine the capacity of a hyperedge and whether or not the hyperedge
 * is relevant for optimizing the objective function.
 */
struct CutFlowNetworkConstruction {
  // ! Capacity of the hyperedge
  template<typename PartitionedHypergraph>
  static HyperedgeWeight capacity(const PartitionedHypergraph& phg,
                                  const Context&,
                                  const HyperedgeID he,
                                  const PartitionID,
                                  const PartitionID) {
    return phg.edgeWeight(he);
  }

  // ! If true, then hyperedge is not relevant and can be dropped.
  template<typename PartitionedHypergraph>
  static bool dropHyperedge(const PartitionedHypergraph& phg,
                            const HyperedgeID he,
                            const PartitionID block_0,
                            const PartitionID block_1) {
    return phg.pinCountInPart(he, block_0) + phg.pinCountInPart(he, block_1) < phg.edgeSize(he);
  }

  // ! If true, then hyperedge is connected to source.
  template<typename PartitionedHypergraph>
  static bool connectToSource(const PartitionedHypergraph&,
                              const HyperedgeID,
                              const PartitionID,
                              const PartitionID) {
    return false;
  }

  // ! If true, then hyperedge is connected to sink.
  template<typename PartitionedHypergraph>
  static bool connectToSink(const PartitionedHypergraph&,
                            const HyperedgeID,
                            const PartitionID,
                            const PartitionID) {
    return false;
  }

  // ! If true, then hyperedge is considered as cut edge and its
  // ! weight is added to the total cut
  template<typename PartitionedHypergraph>
  static bool isCut(const PartitionedHypergraph&,
                    const HyperedgeID,
                    const PartitionID,
                    const PartitionID) {
    return false;
  }
};

}  // namespace mt_kahypar

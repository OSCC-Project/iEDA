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

#include "gmock/gmock.h"

#include "mt-kahypar/datastructures/contraction_tree.h"

namespace mt_kahypar {
namespace ds {

void verifyChilds(const ContractionTree& tree,
                  const HypernodeID u,
                  const std::set<HypernodeID>& expected_childs) {
  size_t num_childs = 0;
  for ( const HypernodeID v : tree.childs(u) ) {
    ASSERT_TRUE(expected_childs.find(v) != expected_childs.end())
      << "Child " << v << " not found!";
    ++num_childs;
  }
  ASSERT_EQ(num_childs, expected_childs.size());
  ASSERT_EQ(num_childs, tree.degree(u));
}

void verifyRoots(const parallel::scalable_vector<HypernodeID>& actual_roots,
                 const std::set<HypernodeID> expected_roots) {
  ASSERT_EQ(expected_roots.size(), actual_roots.size());
  for ( const HypernodeID& u : actual_roots ) {
    ASSERT_TRUE(expected_roots.find(u) != expected_roots.end())
      << "Root " << u << " not found!";
  }
}

TEST(AContractionTree, IsConstructedCorrectly1) {
  ContractionTree tree;
  tree.initialize(5);
  tree.setParent(1, 0);
  tree.setParent(2, 0);
  tree.setParent(3, 1);
  tree.setParent(4, 1);
  tree.finalize();

  verifyChilds(tree, 0, { 1, 2 });
  verifyChilds(tree, 1, { 3, 4 });
  verifyChilds(tree, 2, { });
  verifyChilds(tree, 3, { });
  verifyChilds(tree, 4, { });

  ASSERT_EQ(4, tree.subtreeSize(0));
  ASSERT_EQ(2, tree.subtreeSize(1));
  ASSERT_EQ(0, tree.subtreeSize(2));
  ASSERT_EQ(0, tree.subtreeSize(3));
  ASSERT_EQ(0, tree.subtreeSize(4));

  verifyRoots(tree.roots(), { 0 });
  verifyRoots(tree.roots_of_version(0), { 0 });
}

TEST(AContractionTree, IsConstructedCorrectly2) {
  ContractionTree tree;
  tree.initialize(10);
  tree.setParent(1, 0);
  tree.setParent(2, 0);
  tree.setParent(3, 1);
  tree.setParent(4, 1);
  tree.setParent(6, 5);
  tree.setParent(7, 5);
  tree.setParent(8, 5);
  tree.setParent(9, 5);
  tree.finalize();

  verifyChilds(tree, 0, { 1, 2 });
  verifyChilds(tree, 1, { 3, 4 });
  verifyChilds(tree, 2, { });
  verifyChilds(tree, 3, { });
  verifyChilds(tree, 4, { });
  verifyChilds(tree, 5, { 6, 7, 8, 9 });
  verifyChilds(tree, 6, { });
  verifyChilds(tree, 7, { });
  verifyChilds(tree, 8, { });
  verifyChilds(tree, 9, { });

  ASSERT_EQ(4, tree.subtreeSize(0));
  ASSERT_EQ(2, tree.subtreeSize(1));
  ASSERT_EQ(0, tree.subtreeSize(2));
  ASSERT_EQ(0, tree.subtreeSize(3));
  ASSERT_EQ(0, tree.subtreeSize(4));
  ASSERT_EQ(4, tree.subtreeSize(5));
  ASSERT_EQ(0, tree.subtreeSize(6));
  ASSERT_EQ(0, tree.subtreeSize(7));
  ASSERT_EQ(0, tree.subtreeSize(8));
  ASSERT_EQ(0, tree.subtreeSize(9));

  verifyRoots(tree.roots(), { 0, 5 });
  verifyRoots(tree.roots_of_version(0), { 0, 5 });
}

TEST(AContractionTree, IsConstructedCorrectly3) {
  ContractionTree tree;
  tree.initialize(10);
  tree.setParent(1, 0);
  tree.setParent(2, 0);
  tree.setParent(3, 1);
  tree.setParent(4, 1);
  tree.setParent(5, 2);
  tree.setParent(6, 2);
  tree.setParent(7, 3);
  tree.setParent(8, 3);
  tree.setParent(9, 4);
  tree.finalize();

  verifyChilds(tree, 0, { 1, 2 });
  verifyChilds(tree, 1, { 3, 4 });
  verifyChilds(tree, 2, { 5, 6 });
  verifyChilds(tree, 3, { 7, 8 });
  verifyChilds(tree, 4, { 9 });
  verifyChilds(tree, 5, { });
  verifyChilds(tree, 6, { });
  verifyChilds(tree, 7, { });
  verifyChilds(tree, 8, { });
  verifyChilds(tree, 9, { });

  ASSERT_EQ(9, tree.subtreeSize(0));
  ASSERT_EQ(5, tree.subtreeSize(1));
  ASSERT_EQ(2, tree.subtreeSize(2));
  ASSERT_EQ(2, tree.subtreeSize(3));
  ASSERT_EQ(1, tree.subtreeSize(4));
  ASSERT_EQ(0, tree.subtreeSize(5));
  ASSERT_EQ(0, tree.subtreeSize(6));
  ASSERT_EQ(0, tree.subtreeSize(7));
  ASSERT_EQ(0, tree.subtreeSize(8));
  ASSERT_EQ(0, tree.subtreeSize(9));

  verifyRoots(tree.roots(), { 0 });
  verifyRoots(tree.roots_of_version(0), { 0 });
}

TEST(AContractionTree, IsConstructedCorrectly4) {
  ContractionTree tree;
  tree.initialize(21);
  tree.setParent(1, 0);
  tree.setParent(2, 0);
  tree.setParent(4, 3);
  tree.setParent(5, 3);
  tree.setParent(7, 6);
  tree.setParent(8, 6);
  tree.setParent(10, 9);
  tree.setParent(11, 9);
  tree.setParent(13, 12);
  tree.setParent(14, 12);
  tree.setParent(16, 15);
  tree.setParent(17, 15);
  tree.setParent(19, 18);
  tree.setParent(20, 18);
  tree.finalize();

  verifyChilds(tree, 0, { 1, 2 });
  verifyChilds(tree, 1, { });
  verifyChilds(tree, 2, { });
  verifyChilds(tree, 3, { 4, 5 });
  verifyChilds(tree, 4, { });
  verifyChilds(tree, 5, { });
  verifyChilds(tree, 6, { 7, 8 });
  verifyChilds(tree, 7, { });
  verifyChilds(tree, 8, { });
  verifyChilds(tree, 9, { 10, 11 });
  verifyChilds(tree, 10, { });
  verifyChilds(tree, 11, { });
  verifyChilds(tree, 12, { 13, 14 });
  verifyChilds(tree, 13, { });
  verifyChilds(tree, 14, { });
  verifyChilds(tree, 15, { 16, 17 });
  verifyChilds(tree, 16, { });
  verifyChilds(tree, 17, { });
  verifyChilds(tree, 18, { 19, 20 });
  verifyChilds(tree, 19, { });
  verifyChilds(tree, 20, { });

  ASSERT_EQ(2, tree.subtreeSize(0));
  ASSERT_EQ(0, tree.subtreeSize(1));
  ASSERT_EQ(0, tree.subtreeSize(2));
  ASSERT_EQ(2, tree.subtreeSize(3));
  ASSERT_EQ(0, tree.subtreeSize(4));
  ASSERT_EQ(0, tree.subtreeSize(5));
  ASSERT_EQ(2, tree.subtreeSize(6));
  ASSERT_EQ(0, tree.subtreeSize(7));
  ASSERT_EQ(0, tree.subtreeSize(8));
  ASSERT_EQ(2, tree.subtreeSize(9));
  ASSERT_EQ(0, tree.subtreeSize(10));
  ASSERT_EQ(0, tree.subtreeSize(11));
  ASSERT_EQ(2, tree.subtreeSize(12));
  ASSERT_EQ(0, tree.subtreeSize(13));
  ASSERT_EQ(0, tree.subtreeSize(14));
  ASSERT_EQ(2, tree.subtreeSize(15));
  ASSERT_EQ(0, tree.subtreeSize(16));
  ASSERT_EQ(0, tree.subtreeSize(17));
  ASSERT_EQ(2, tree.subtreeSize(18));
  ASSERT_EQ(0, tree.subtreeSize(19));
  ASSERT_EQ(0, tree.subtreeSize(20));

  verifyRoots(tree.roots(), { 0, 3, 6, 9, 12, 15, 18 });
  verifyRoots(tree.roots_of_version(0), { 0, 3, 6, 9, 12, 15, 18 });
}

TEST(AContractionTree, ContainsCorrectRootsInPresenceOfSingletonRoots) {
  ContractionTree tree;
  tree.initialize(10);
  tree.setParent(1, 0);
  tree.setParent(2, 0);
  tree.finalize();

  verifyRoots(tree.roots(), { 0 });
  verifyRoots(tree.roots_of_version(0), { 0 });
}

void verifyChildsOfVersion(const ContractionTree& tree,
                           const HypernodeID u,
                           const size_t version,
                           const std::set<HypernodeID>& expected_childs) {
  size_t num_childs = 0;
  tree.doForEachChildOfVersion(u, version, [&](const HypernodeID& v) {
    ASSERT_TRUE(expected_childs.find(v) != expected_childs.end())
      << "Node " << v << " not contained in expected childs of node " << u
      << " for version " << version;
    ++num_childs;
  });
  ASSERT_EQ(num_childs, expected_childs.size());
}

TEST(AContractionTree, ContainsCorrectVersionRoots1) {
  ContractionTree tree;
  tree.initialize(5);
  tree.setParent(1, 0, 1);
  tree.setParent(2, 0, 1);
  tree.setParent(3, 1, 0);
  tree.setParent(4, 1, 0);
  tree.finalize(2);

  verifyRoots(tree.roots(), { 0 });
  verifyRoots(tree.roots_of_version(0), { 1 });
  verifyRoots(tree.roots_of_version(1), { 0 });
  verifyChildsOfVersion(tree, 0, 0, { });
  verifyChildsOfVersion(tree, 0, 1, { 1, 2 });
  verifyChildsOfVersion(tree, 1, 0, { 3, 4 });
  verifyChildsOfVersion(tree, 1, 1, { });
}

TEST(AContractionTree, ContainsCorrectVersionRoots2) {
  ContractionTree tree;
  tree.initialize(7);
  tree.setParent(1, 0, 2);
  tree.setParent(2, 0, 1);
  tree.setParent(3, 1, 0);
  tree.setParent(4, 1, 1);
  tree.setParent(5, 2, 0);
  tree.setParent(6, 2, 1);
  tree.finalize(3);

  verifyRoots(tree.roots(), { 0 });
  verifyRoots(tree.roots_of_version(0), { 1, 2 });
  verifyRoots(tree.roots_of_version(1), { 0, 1 });
  verifyRoots(tree.roots_of_version(2), { 0 });
  verifyChildsOfVersion(tree, 0, 1, { 2 });
  verifyChildsOfVersion(tree, 0, 2, { 1 });
  verifyChildsOfVersion(tree, 1, 0, { 3 });
  verifyChildsOfVersion(tree, 1, 1, { 4 });
  verifyChildsOfVersion(tree, 2, 0, { 5 });
  verifyChildsOfVersion(tree, 2, 1, { 6 });
}

TEST(AContractionTree, ContainsCorrectVersionRoots3) {
  ContractionTree tree;
  tree.initialize(6);
  tree.setParent(1, 0, 2);
  tree.setParent(2, 0, 1);
  tree.setParent(4, 3, 0);
  tree.setParent(5, 3, 1);
  tree.finalize(3);

  verifyRoots(tree.roots(), { 0, 3 });
  verifyRoots(tree.roots_of_version(0), { 3 });
  verifyRoots(tree.roots_of_version(1), { 0, 3 });
  verifyRoots(tree.roots_of_version(2), { 0 });
  verifyChildsOfVersion(tree, 0, 1, { 2 });
  verifyChildsOfVersion(tree, 0, 2, { 1 });
  verifyChildsOfVersion(tree, 3, 0, { 4 });
  verifyChildsOfVersion(tree, 3, 1, { 5 });
}

TEST(AContractionTree, ContainsCorrectVersionRoots4) {
  ContractionTree tree;
  tree.initialize(10);
  tree.setParent(1, 0, 4);
  tree.setParent(2, 0, 4);
  tree.setParent(3, 1, 1);
  tree.setParent(4, 2, 2);
  tree.setParent(6, 5, 3);
  tree.setParent(7, 5, 4);
  tree.setParent(8, 6, 0);
  tree.setParent(9, 7, 2);
  tree.finalize(5);

  verifyRoots(tree.roots(), { 0, 5 });
  verifyRoots(tree.roots_of_version(0), { 6 });
  verifyRoots(tree.roots_of_version(1), { 1 });
  verifyRoots(tree.roots_of_version(2), { 2, 7 });
  verifyRoots(tree.roots_of_version(3), { 5 });
  verifyRoots(tree.roots_of_version(4), { 0, 5 });
  verifyChildsOfVersion(tree, 0, 4, { 1, 2 });
  verifyChildsOfVersion(tree, 1, 1, { 3 });
  verifyChildsOfVersion(tree, 2, 2, { 4 });
  verifyChildsOfVersion(tree, 5, 3, { 6 });
  verifyChildsOfVersion(tree, 5, 4, { 7 });
  verifyChildsOfVersion(tree, 6, 0, { 8 });
  verifyChildsOfVersion(tree, 7, 2, { 9 });
}


} // namespace ds
} // namespace mt_kahypar
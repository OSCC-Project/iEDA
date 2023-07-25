#pragma once

#include "tree.h"

namespace salt {

// O(n log n) impl of MST building
// 1. spanning graph and m = O(n) edges in O(n log n) time
// 2. Prim's algorithm in O(m log m) = O(n log n) time
class MstBuilder {
public:
    void Run(const Net& net, Tree& tree);

    // There are some redundancy in adjLists
    void GetAllNearestNeighbors(const vector<Point>& points, vector<vector<int>>& adjLists);

private:
    void GetSpanGraph(const Net& net, vector<vector<int>>& adjLists);
    void RunPrimAlg(const Net& net, const vector<vector<int>>& adjLists, Tree& tree);

    // work in place
    // handle(i, j): j is the octant-NN of i
    // close on both boundaries (the clockwise and counter-clockwise ones)
    void GetNearestNeighbors(int maxOctant, vector<pair<Point, int>>& points, function<void(int, int)> handle);
    void GetFirstOctantNearestNeighbors(vector<pair<Point, int>>& points, function<void(int, int)> handle);
};

}  // namespace salt
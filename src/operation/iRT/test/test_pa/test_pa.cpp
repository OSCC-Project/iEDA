#include <algorithm>
#include <climits>
#include <cmath>
#include <iostream>
#include <vector>

using namespace std;

struct Point
{
  int x, y;
};

int manhattanDistance(const Point& p1, const Point& p2)
{
  return abs(p1.x - p2.x) + abs(p1.y - p2.y);
}

int minDistance(const vector<Point>& points)
{
  int minDist = INT_MAX;
  for (size_t i = 0; i < points.size(); ++i) {
    for (size_t j = i + 1; j < points.size(); ++j) {
      int dist = manhattanDistance(points[i], points[j]);
      minDist = min(minDist, dist);
    }
  }
  return minDist;
}

void branchAndBound(const vector<vector<Point>>& groups, vector<Point>& currentCombination, vector<Point>& bestCombination,
                    int& bestMinDist, size_t groupIndex)
{
  if (groupIndex == groups.size()) {
    int currentMinDist = minDistance(currentCombination);
    if (currentMinDist > bestMinDist) {
      bestMinDist = currentMinDist;
      bestCombination = currentCombination;
    }
    return;
  }

  for (const auto& point : groups[groupIndex]) {
    currentCombination.push_back(point);
    if (groupIndex == 0 || minDistance(currentCombination) > bestMinDist) {
      branchAndBound(groups, currentCombination, bestCombination, bestMinDist, groupIndex + 1);
    }
    currentCombination.pop_back();
  }
}

vector<Point> selectPoints(const vector<vector<Point>>& groups)
{
  vector<Point> bestCombination;
  vector<Point> currentCombination;
  int bestMinDist = -1;

  branchAndBound(groups, currentCombination, bestCombination, bestMinDist, 0);

  return bestCombination;
}

int main()
{
  vector<vector<Point>> groups = {{{1, 2}, {3, 4}, {5, 6}}, {{7, 8}, {9, 10}, {11, 12}}, {{13, 14}, {15, 16}, {17, 18}}};

  vector<Point> bestCombination = selectPoints(groups);

  cout << "Selected points:" << endl;
  for (const auto& p : bestCombination) {
    cout << "(" << p.x << ", " << p.y << ")" << endl;
  }

  return 0;
}

#ifndef IDRC_SRC_DB_SPACING_RANGE_WIDTH_H_
#define IDRC_SRC_DB_SPACING_RANGE_WIDTH_H_
#include <algorithm>
#include <utility>
namespace idrc {
class SpacingRangeRule
{
 public:
  SpacingRangeRule() {}
  SpacingRangeRule(const SpacingRangeRule& other)
  {
    _min_width = other._min_width;
    _max_width = other._max_width;
    _spacing = other._spacing;
  }
  SpacingRangeRule(SpacingRangeRule&& other)
  {
    _min_width = std::move(other._min_width);
    _max_width = std::move(other._max_width);
    _spacing = std::move(other._spacing);
  }
  ~SpacingRangeRule() {}
  SpacingRangeRule& operator=(const SpacingRangeRule& other)
  {
    _min_width = other._min_width;
    _max_width = other._max_width;
    _spacing = other._spacing;
    return *this;
  }
  SpacingRangeRule& operator=(SpacingRangeRule&& other)
  {
    _min_width = std::move(other._min_width);
    _max_width = std::move(other._max_width);
    _spacing = std::move(other._spacing);
    return *this;
  }
  // setter
  void set_min_width(int min_width) { _min_width = min_width; }
  void set_max_width(int max_width) { _max_width = max_width; }
  void set_spacing(int spacing) { _spacing = spacing; }
  // getter
  int get_min_width() const { return _min_width; }
  int get_max_width() const { return _max_width; }
  int get_spacing() const { return _spacing; }

 private:
  int _min_width = 0;
  int _max_width = 0;
  int _spacing = 0;
};
}  // namespace idrc

#endif
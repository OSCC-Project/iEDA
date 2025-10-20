/**
 * @file geometry.hh
 * @author Fuxing Huang (fxxhuang@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-06-19
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef IPL_GEOMETRY_HH
#define IPL_GEOMETRY_HH
#include <vector>
using std::size_t;
using std::vector;
namespace ipl {
template <typename T>
class Coordinate
{
 public:
  Coordinate() : _x{0}, _y{0} {}
  Coordinate(T x, T y) : _x(x), _y(y) {}
  constexpr Coordinate<T> operator*(const T& b) const { return {_x * b, _y * b}; }
  constexpr Coordinate<T> operator+(const Coordinate<T>& other) const { return {_x + other._x, _y + other._y}; }
  constexpr Coordinate<T> operator-(const Coordinate<T>& other) const { return {_x - other._x, _y - other._y}; }
  constexpr bool operator==(const Coordinate& other) const { return _x == other._x && _y == other._y; }
  constexpr bool operator<(const Coordinate& other) const { return _x < other._x || (_x == other._x && _y < other._y); }
  constexpr bool operator>(const Coordinate& other) const { return other < *this; }
  constexpr bool operator<=(const Coordinate& other) const { return !(other < *this); }
  constexpr bool operator>=(const Coordinate& other) const { return !(*this < other); }
  constexpr Coordinate<T> operator+=(const Coordinate<T>& other)
  {
    _x += other._x;
    _y += other._y;
    return *this;
  }
  constexpr Coordinate<T> operator-=(const Coordinate<T>& other)
  {
    _x -= other._x;
    _y -= other._y;
    return *this;
  }
  constexpr Coordinate<T> operator*=(const T& b)
  {
    _x *= b;
    _y *= b;
    return *this;
  }
  static constexpr T dot(const Coordinate<T>& lhs, const Coordinate<T>& rhs) { return lhs._x * rhs._x + lhs._y * rhs._y; }
  static constexpr T cross(const Coordinate<T>& lhs, const Coordinate<T>& rhs) { return lhs._x * rhs._y - rhs._x * lhs._y; }
  constexpr T get_x() const { return _x; }
  constexpr T get_y() const { return _y; }

 private:
  T _x;
  T _y;
};
template <typename T>
struct CoordinateHash
{
  std::size_t operator()(const Coordinate<T>& p) const noexcept
  {
    std::size_t h1 = std::hash<T>{}(p.get_x());
    std::size_t h2 = std::hash<T>{}(p.get_y());
    return h1 ^ (h2 << 1);
  }
};

template <typename T>
class Line
{
 public:
  Line() : _a{}, _b{} {}
  Line(const std::initializer_list<Coordinate<T>>& args) : _a{*args.begin()}, _b(*(args.begin() + 1)) { init(); }
  Line(const Coordinate<T>& a, const Coordinate<T>& b) : _a{a}, _b{b} { init(); }
  bool isLeftPoint(const Coordinate<T>& coordinate) const { return Coordinate<T>::cross(_dis, coordinate - _b) < T{0}; }
  bool isOnlinePoint(const Coordinate<T>& coordinate) const { return Coordinate<T>::cross(_dis, coordinate - _a) == T{0}; }
  Coordinate<T> operator&(const Line<T>& other) const { return intersection(*this, other); }
  Coordinate<T> operator&(Line<T>&& other) const { return intersection(*this, other); }
  static constexpr Coordinate<T> intersection(const Line<T>& lhs, const Line<T>& rhs)
  {
    return (rhs._dis * lhs._cross - lhs._dis * rhs._cross) * (T{1} / Coordinate<T>::cross(lhs._dis, rhs._dis));
  }

 private:
  void init()
  {
    _dis = {_a - _b};
    _cross = Coordinate<T>::cross(_a, _b);
  }

 private:
  Coordinate<T> _a;
  Coordinate<T> _b;
  Coordinate<T> _dis;
  T _cross;
};

template <typename T>
class Ploygon
{
 public:
  Ploygon() {}
  Ploygon(vector<Coordinate<T>>&& coordis) : _coordis(coordis) {}
  Ploygon(const std::initializer_list<Coordinate<T>>& args) : _coordis(args) {}
  ~Ploygon() {}
  bool isEmpty() const { return _coordis.empty(); }
  constexpr Ploygon<T> operator&(const Ploygon<T>& other) const { return PloygonClip(*this, other); }
  // constexpr Ploygon<T> operator&(vector<Coordinate<T>>&& other) const { return PloygonClip(*this, Ploygon<T>(other)); }
  constexpr T area() const
  {
    T ans{0};
    if (isEmpty())
      return ans;
    for (size_t i = 1; i < _coordis.size() - 1; i++) {
      ans += Coordinate<T>::cross(_coordis[i] - _coordis[0], _coordis[i + 1] - _coordis[0]);
    }
    return ans / 2;
  }
  vector<Coordinate<T>> getCoordinates() const { return _coordis; }

 private:
  Ploygon<T> PloygonClip(const Ploygon<T>& lhs, const Ploygon<T>& rhs) const;

 private:
  vector<Coordinate<T>> _coordis;
};
template <typename T>
Ploygon<T> Ploygon<T>::PloygonClip(const Ploygon<T>& lhs, const Ploygon<T>& rhs) const
{
  if (lhs.isEmpty() || rhs.isEmpty())
    return Ploygon<T>();
  std::vector<Coordinate<T>> ring = {lhs._coordis.begin(), lhs._coordis.end()};

  Coordinate<T> p1 = *rhs._coordis.rbegin();

  std::vector<Coordinate<T>> input;

  for (Coordinate<T> p2 : rhs._coordis) {
    input.clear();
    input.insert(input.end(), ring.begin(), ring.end());
    Coordinate<T> s = input[input.size() - 1];

    ring.clear();

    for (Coordinate<T> e : input) {
      Line<T> line{p1, p2};

      if (line.isLeftPoint(e)) {
        if (!line.isLeftPoint(s)) {
          ring.push_back(line & Line<T>(s, e));
        }

        ring.push_back(e);
      } else if (line.isLeftPoint(s)) {
        ring.push_back(line & Line<T>(s, e));
      }
      s = e;
    }

    p1 = p2;
  }

  return Ploygon<T>(std::move(ring));
}
}  // namespace ipl

#endif

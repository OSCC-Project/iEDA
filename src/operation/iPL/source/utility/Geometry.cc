#include "Geometry.hh"
namespace ipl {
// template <typename T>

// template <typename T>
// Ploygon<T>::Ploygon()
// {
// }

// template <typename T>
// Ploygon<T> Ploygon<T>::operator&(const Ploygon<T>& other)
// {
//   if (isEmpty() || other.isEmpty())
//     return {};
//   std::vector<Coordinate<T>> ring{_coordis.begin(), _coordis.end()};

//   Coordinate<T> p1 = *other._coodis.rbegin();

//   std::vector<Coordinate<T>> input;

//   for (Coordinate<T> p2 : other._coodis) {
//     input.clear();
//     input.insert(input.end(), ring.begin(), ring.end());
//     Coordinate<T> s = input[input.size() - 1];

//     ring.clear();

//     for (Coordinate<T> e : input) {
//       Line<T> line{p1, p2};

//       if (line.isLeftPoint(e)) {
//         if (!line.isLeftPoint(s)) {
//           ring.push_back(line & (s, e));
//         }

//         ring.push_back(e);
//       } else if (line.isLeftPoint(s)) {
//         ring.push_back(line & (s, e));
//       }
//       s = e;
//     }

//     p1 = p2;
//   }

//   return ring;
// }

}  // namespace ipl


#include <cassert>
#include <deque>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <valarray>
#include <vector>

extern "C" {
#include "zlib.h"
}

/*!
@file VCDTypes.hpp
@brief A file for common types and data structures used by the VCD parser.
*/

#ifndef VCDTypes_HPP
#define VCDTypes_HPP

//! Friendly name for a signal
typedef std::string VCDSignalReference;

//! Friendly name for a scope
typedef std::string VCDScopeName;

//! Compressed hash representation of a signal.
typedef std::string VCDSignalHash;

//! Specifies the timing resoloution along with VCDTimeUnit
typedef unsigned VCDTimeRes;

//! Width in bits of a signal.
typedef unsigned VCDSignalSize;

//! Represents the four-state signal values of a VCD file.
enum VCDBit : char
{
  VCD_0 = 0,  //!< Logic zero
  VCD_1 = 1,  //!< Logic one
  VCD_X = 2,  //!< Unknown / Undefined
  VCD_Z = 3   //!< High Impedence.
};

#ifdef COMPRESS_BIT

using VCDCompressBit = unsigned char;

constexpr unsigned char LOW_BITS = 0B00000011;
constexpr unsigned char SECOND_BITS = 0B00001100;
constexpr unsigned char THIRD_BITS = 0B00110000;
constexpr unsigned char HIGH_BITS = 0B11000000;

class VCDBitVector
{
 public:
  explicit VCDBitVector(std::size_t bit_vec_size) : _bit_vec((bit_vec_size & 0x03) ? (((bit_vec_size >> 2) + 1) << 2) : bit_vec_size) {}
  ~VCDBitVector() = default;

  auto decode_position(std::size_t pos)
  {
    std::size_t compress_pos = pos >> 2;
    int compress_inner_pos = pos & 3L;

    return std::make_pair(compress_pos, compress_inner_pos);
  }
  VCDBit operator[](std::size_t pos)
  {
    auto [compress_pos, compress_inner_pos] = decode_position(pos);
    auto& the_bits = _bit_vec[compress_pos];

    static std::unordered_map<int, std::pair<unsigned char, int>> bit_to_mask_shift
        = {{0, {LOW_BITS, 0}}, {1, {SECOND_BITS, 2}}, {2, {THIRD_BITS, 4}}, {3, {HIGH_BITS, 6}}};
    auto [mask, shift] = bit_to_mask_shift[compress_inner_pos];

    return (VCDBit)((the_bits & mask) >> shift);
  }

  void setValue(std::size_t pos, VCDBit bit_value)
  {
    auto [compress_pos, compress_inner_pos] = decode_position(pos);
    auto& the_bits = _bit_vec[compress_pos];

    the_bits |= ((char) bit_value << (compress_inner_pos << 1));
  }

  auto& get_bit_vec() { return _bit_vec; }

  void compressBitVec()
  {
    if (!c_zlib_compress) {
      return;
    }

    if (_bit_vec.size() < c_compress_bit_size) {
      return;
    }

    auto* buffer = new unsigned char[_bit_vec.size()];
    unsigned long compr_len = _bit_vec.size();
    if (Z_OK == compress(buffer, &compr_len, &(_bit_vec[0]), _bit_vec.size())) {
      std::valarray<VCDCompressBit> compress_bit_vec(buffer, compr_len);
      std::swap(_bit_vec, compress_bit_vec);
      _is_zlib_compress = true;
    }

    delete[] buffer;
  }

  void uncompressBitVec()
  {
    if (!c_zlib_compress || !_is_zlib_compress) {
      return;
    }

    auto* buffer = new unsigned char[_bit_vec.size()];
    unsigned long uncompr_len = _bit_vec.size();
    if (Z_OK == uncompress(buffer, &uncompr_len, &(_bit_vec[0]), _bit_vec.size())) {
      std::valarray<VCDCompressBit> uncompress_bit_vec(buffer, uncompr_len);
      std::swap(_bit_vec, uncompress_bit_vec);
    } else {
      assert(0);
    }

    delete[] buffer;
  }

 private:
  std::valarray<VCDCompressBit> _bit_vec;
  bool _is_zlib_compress = false;
};

#else
//! A vector of VCDBit values.
typedef std::valarray<VCDBit> VCDBitVector;
#endif

//! Typedef to identify a real number as stored in a VCD.
typedef double VCDReal;

//! Describes how a signal value is represented in the VCD trace.
typedef enum
{
  VCD_SCALAR,  //!< Single VCDBit
  VCD_VECTOR,  //!< Vector of VCDBit
  VCD_REAL     //!< IEEE Floating point (64bit).
} VCDValueType;

class VCDTimedValue;

//! A vector of tagged time/value pairs, sorted by time values.
typedef std::deque<VCDTimedValue> VCDSignalValues;

//! Variable types of a signal in a VCD file.
typedef enum
{
  VCD_VAR_EVENT,
  VCD_VAR_INTEGER,
  VCD_VAR_PARAMETER,
  VCD_VAR_REAL,
  VCD_VAR_REALTIME,
  VCD_VAR_REG,
  VCD_VAR_SUPPLY0,
  VCD_VAR_SUPPLY1,
  VCD_VAR_TIME,
  VCD_VAR_TRI,
  VCD_VAR_TRIAND,
  VCD_VAR_TRIOR,
  VCD_VAR_TRIREG,
  VCD_VAR_TRI0,
  VCD_VAR_TRI1,
  VCD_VAR_WAND,
  VCD_VAR_WIRE,
  VCD_VAR_WOR
} VCDVarType;

//! Represents the possible time units a VCD file is specified in.
typedef enum
{
  TIME_S,   //!< Seconds
  TIME_MS,  //!< Milliseconds
  TIME_US,  //!< Microseconds
  TIME_NS,  //!< Nanoseconds
  TIME_PS,  //!< Picoseconds
  TIME_FS,  //!< Femtoseconds
} VCDTimeUnit;

//! Represents the type of SV construct who's scope we are in.
typedef enum
{
  VCD_SCOPE_BEGIN,
  VCD_SCOPE_FORK,
  VCD_SCOPE_FUNCTION,
  VCD_SCOPE_MODULE,
  VCD_SCOPE_TASK,
  VCD_SCOPE_ROOT
} VCDScopeType;

// Typedef over vcdscope to make it available to VCDSignal struct.
class VCDScope;
//! Represents a single signal reference within a VCD file
struct VCDSignal
{
  VCDSignalHash hash;
  VCDSignalReference reference;
  VCDScope* scope;
  VCDSignalSize size;
  VCDVarType type;
  int lindex;  // -1 if no brackets, otherwise [lindex] or [lindex:rindex]
  int rindex;  // -1 if not [lindex:rindex]
};

//! Represents a scope type, scope name pair and all of it's child signals.
struct VCDScope
{
  VCDScopeName name;                //!< The short name of the scope
  VCDScopeType type;                //!< Construct type
  VCDScope* parent;                 //!< Parent scope object
  std::vector<VCDScope*> children;  //!< Child scope objects.
  std::vector<VCDSignal*> signals;  //!< Signals in this scope.
};

#endif

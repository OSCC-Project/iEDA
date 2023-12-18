
#ifndef VCDValue_HPP
#define VCDValue_HPP

#include <iostream>
#include <variant>

#include "VCDTypes.hpp"
#include "log/Log.hh"

/*!
@brief Represents a single value found in a VCD File.
@details Can contain a single bit (a scalar), a bti vector, or an
IEEE floating point number.
*/
class VCDValue {
  //! Convert a VCDBit to a single char
  static char VCDBit2Char(VCDBit b) {
    switch (b) {
      case (VCD_0):
        return '0';
      case (VCD_1):
        return '1';
      case (VCD_Z):
        return 'Z';
      case (VCD_X):
      default:
        return 'X';
    }
  }

 public:
  VCDValue() = default;
  /*!
  @brief Create a new VCDValue with the type VCD_SCALAR
  */
  VCDValue(VCDBit value) { this->value = value; }

  /*!
  @brief Create a new VCDValue with the type VCD_VECTOR
  */
  VCDValue(VCDBitVector&& value) { this->value = std::move(value); }

  /*!
  @brief Create a new VCDValue with the type VCD_VECTOR
  */
  VCDValue(VCDReal value) { this->value = value; }

  ~VCDValue() = default;

  VCDValue(const VCDValue& orig) = delete;

  VCDValue& operator=(const VCDValue& orig) = delete;

  VCDValue(VCDValue&& other) noexcept { value = std::move(other.value); }

  VCDValue& operator=(VCDValue&& other) noexcept {
    value = std::move(other.value);

    return *this;
  }

  // void* operator new(size_t size);
  // void operator delete(void* ptr);

  //! Return the type of value stored by this class instance.
  VCDValueType get_type() { return VCDValueType(value.index()); }

  //! Get the bit value of the instance.
  VCDBit get_value_bit() {
    LOG_FATAL_IF(get_type() != VCD_SCALAR) << "value type error";
    return std::get<0>(value);
  }

  //! Get the vector value of the instance.
  VCDBitVector& get_value_vector() {
    LOG_FATAL_IF(get_type() != VCD_VECTOR) << "value type error";
    return std::get<1>(value);
  }

  //! Get the real value of the instance.
  VCDReal get_value_real() {
    LOG_FATAL_IF(get_type() != VCD_REAL) << "value type error";
    return std::get<2>(value);
  }

 protected:
  //! The actual value stored, as identified by type.
  std::variant<VCDBit, VCDBitVector, VCDReal> value;
};

//! Represents a single instant in time in a trace
typedef int64_t VCDTime;

//! A signal value tagged with times.
class VCDTimedValue {
 public:
  VCDTimedValue() = default;
  ~VCDTimedValue() = default;

  VCDTimedValue(VCDTimedValue&& other) = default;
  VCDTimedValue& operator=(VCDTimedValue&& other) = default;

  VCDTime time;
  VCDValue value;
};

#endif

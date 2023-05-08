#pragma once

namespace idb {

// external library flag
class GdsELFlags
{
 public:
  struct FlagsBits
  {
    unsigned unused : 14;
    unsigned is_external : 1;
    unsigned is_template : 1;
  };

  union FlagsValue
  {
    uint16_t value = 0;
    FlagsBits bits;
  };

  // constructor
  GdsELFlags() : flag() {}

  // getter
  bool is_external() const;
  bool is_template() const;
  auto get_value() const;

  // setter
  void set_is_external(bool);
  void set_is_template(bool);

  // function
  void reset();

  // members
  FlagsValue flag;
};

//////// inline //////

inline bool GdsELFlags::is_external() const
{
  return flag.bits.is_external;
}

inline bool GdsELFlags::is_template() const
{
  return flag.bits.is_template;
}

inline void GdsELFlags::set_is_external(bool v)
{
  flag.bits.is_external = v;
}

inline void GdsELFlags::set_is_template(bool v)
{
  flag.bits.is_template = v;
}

inline void GdsELFlags::reset()
{
  flag.value = 0;
}

inline auto GdsELFlags::get_value() const
{
  return flag.value;
}

}  // namespace idb

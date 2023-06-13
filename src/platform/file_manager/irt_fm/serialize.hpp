#pragma once
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/variant.hpp>
#include <boost/serialization/vector.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <type_traits>

namespace iplf {

template <typename BoostArchive, typename... Args>
void Archive(BoostArchive& ar, Args&... arg)
{
  ((ar & arg), ...);
}

class Timer
{
 public:
  Timer() : start_time(std::chrono::high_resolution_clock::now()) {}

  double elapsed() const
  {
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end_time - start_time;
    return elapsed.count();
  }

 private:
  std::chrono::high_resolution_clock::time_point start_time;
};

template <typename T>
concept InEqualable = requires(T a, T b)
{
  {
    a != b
    } -> std::convertible_to<bool>;
};

template <typename... Args>
class Persister
{
 public:
  Persister(const std::string& path, const std::string& signature = "serializeok") : _path(path), _signature(signature) {}
  using iarchive = boost::archive::binary_iarchive;
  using oarchive = boost::archive::binary_oarchive;

  void save(const Args&... args)
  {
    Timer timer;
    oarchive oar = getOarchive();
    Archive(oar, args...);
    oar << _signature;
    std::cout << "Serialize timecost: " << timer.elapsed() << " ms\n";
  }
  template <InEqualable T>
  void saveWithHeader(const T& header, Args&... args)
  {
    auto oar = getOarchive();
    oar << header;
    Archive(oar, args...);
    oar << _signature;
  }
  template <InEqualable T>
  void loadWithHeader(const T& header, Args&... args)
  {
    auto iar = getIarchive();
    T loaded;
    iar >> loaded;
    if (loaded != header) {
      throw std::runtime_error("Invalid serialize data");
    }
    iplf::Archive(iar, args...);
    std::string sig;
    iar >> sig;
    if (sig != _signature) {
      throw std::runtime_error("Invalid serialize data");
    }
  }
  
  template<InEqualable T>
  T loadHeader(){
    T header;
    auto iar = getIarchive();
    iar >> header;
    return header;
  }
  void load(Args&... args)
  {
    Timer timer;

    iarchive iar = getIarchive();
    Archive(iar, args...);
    std::string sig;

    iar >> sig;
    if (sig != _signature) {
      throw std::runtime_error("Invalid serialize data.");
    }
    std::cout << "Deserialize timecost: " << timer.elapsed() << " ms\n";
  }

  oarchive getOarchive()
  {
    if (!ofs.is_open()) {
      std::filesystem::path dirpath = std::filesystem::path(_path).parent_path();
      if (!std::filesystem::exists(dirpath)) {
        std::filesystem::create_directories(dirpath);
      }
      ofs.open(_path, std::ios::trunc | std::ios::out);
    }
    if (!ofs.is_open()) {
      std::stringstream ss;
      ss << "Failed to open file for writing: " << _path;
      throw std::runtime_error(ss.str());
    }
    return oarchive(ofs);
  }
  iarchive getIarchive()
  {
    if (!ifs.is_open()) {
      ifs.open(_path, std::ios::in);
    }
    if (!ifs.is_open()) {
      std::stringstream ss;
      ss << "Failed to open file for reading: " << _path;
      throw std::runtime_error(ss.str());
    }
    ifs.seekg(0, std::ios::beg);
    return iarchive(ifs);
  }

 private:
  std::string _path;
  std::string _signature;
  std::ifstream ifs;
  std::ofstream ofs;
};

}  // namespace iplf

namespace boost::serialization {

template <typename Serializable, typename Archive>
concept iEDALoadSavable = requires(Serializable& obj, Archive& ar, const unsigned int version)
{
  {
    load(ar, obj, version)
    } -> std::same_as<void>;
  {
    save(ar, obj, version)
    } -> std::same_as<void>;
};

template <typename Serializable, typename Archive>
concept iEDASerializable = !iEDALoadSavable<Serializable, Archive>;

template <typename Archive, typename T>
requires iEDALoadSavable<T, Archive>
void serialize(Archive& ar, T& t, const unsigned int version)
{
  //   using namespace iplf;
  if constexpr (Archive::is_loading::value) {
    load(ar, t, version);
  } else if constexpr (Archive::is_saving::value) {
    save(ar, t, version);
  } else {
    assert(0);
  }
}

template <typename Archive, typename... Types>
void serialize(Archive &ar, std::tuple<Types...> &t, const unsigned int)
{
    std::apply([&](auto &...element)
                { ((ar & element), ...); },
                t);
}

}  // namespace boost::serialization

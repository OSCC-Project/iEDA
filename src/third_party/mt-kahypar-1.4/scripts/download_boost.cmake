include(FetchContent)
FetchContent_Populate(
  boost-src
  URL http://downloads.sourceforge.net/project/boost/boost/1.69.0/boost_1_69_0.tar.bz2
  URL_HASH MD5=a1332494397bf48332cb152abfefcec2
  SOURCE_DIR external_tools/boost
)

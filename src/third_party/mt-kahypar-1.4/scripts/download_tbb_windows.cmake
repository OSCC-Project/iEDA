include(FetchContent)
FetchContent_Populate(
  tbb
  URL https://github.com/oneapi-src/oneTBB/releases/download/v2021.8.0/oneapi-tbb-2021.8.0-win.zip
  URL_HASH SHA256=b9265d4dc5b74e27176c6a6b696882935f605191d014a62c010c9610904e7f65  
  SOURCE_DIR external_tools/tbb
)
#!/bin/bash

./scripts/generate_single_include.sh

for example in $(find example -maxdepth 1 -type f -name '*.cpp'); do
  awk 'BEGIN { count1=0; count2=0 } { if ($0 ~ /^#include \"cx\//) { if (!count1) { print "#include \"cx.hpp\""; count1++ } } else if ($0 ~ /^#include \"wildcards\//) { if (!count2) { print "#include \"wildcards.hpp\""; count2++ } } else { print $0 } }' "$example" > wandbox/"$(basename "$example")"
done

POS=1
for example in $(find wandbox -maxdepth 1 -type f -name '*.cpp' | sort); do
  if [[ "$(basename "$example")" == "example05.cpp" ]]; then
    EXAMPLE_URL=$(scripts/send_to_wandbox.py single_include "$example" gcc-7.2.0 c++17 -- $'-Wall\n-Wextra\n-lstdc++fs' | sed -e 's/.*\(http:\/\/[^ '"'"']*\).*/\1/')
  else
    EXAMPLE_URL=$(scripts/send_to_wandbox.py single_include "$example" gcc-7.2.0 c++11 -- $'-Wall\n-Wextra' | sed -e 's/.*\(http:\/\/[^ '"'"']*\).*/\1/')
  fi
  echo "$example: $EXAMPLE_URL"
  awk -v pos=$POS '/## Example/{ c+=1 } { if (c == pos) { sub("\(https://wandbox.org/permlink/.+\)", "('"$EXAMPLE_URL"')", $0) }; print $0 }' example/README.md > example/README_tmp.md && mv example/README_tmp.md example/README.md
  ((POS++))
done

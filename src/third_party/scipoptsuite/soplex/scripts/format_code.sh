#!/bin/bash

# Use this script to check the format of your code and format
# your code automatically according to astyle rules defined for soplex.
#
# Usage as a user:
#        ./scripts/format_code.sh
#
# Usage for automatic checking (i.e. in ci): returns 0 if everything fine, 1 otherwise
#        ONLY_CHECK_ASTYLE=true ./format_code.sh

FILEPATTERNS="src/WILDCARD.h src/WILDCARD.hpp src/WILDCARD.cpp src/soplex/WILDCARD.h src/soplex/WILDCARD.hpp src/soplex/WILDCARD.cpp"
GITFILES=$(echo "${FILEPATTERNS//WILDCARD/\*}")
ASTYLEFILES=$(echo "${FILEPATTERNS//WILDCARD/\\*}")

astyle --dry-run --options=astylecfg --recursive $ASTYLEFILES > astyleoutput.log
NFORMATTED=$(cat astyleoutput.log | grep ^Formatted | wc -l)

if [ "$NFORMATTED" != "0" ]; then
    echo ">>> Code does not comply with astyle rules, found errors in the following files:"
    echo -e "$(grep ^Formatted astyleoutput.log)"

  # option for automatic checking in ci
  if [ "$ONLY_CHECK_ASTYLE" == "true" ]; then
      exit 1
  fi
  rm astyleoutput.log

  read -p ">>> Format code automatically? [y|N]" -n 2 -r
  echo    # (optional) move to a new line
  if [[ $REPLY =~ ^[Yy]$ ]]; then
      # do dangerous stuff

      ISCLEAN=$(git diff --stat -- $GITFILES |wc -l)
      if [ "$ISCLEAN" == "0" ]; then
          echo ">>> No unstaged changes. Formatting inplace."
          astyle -n --options=astylecfg --recursive $ASTYLEFILES > /dev/null
      else
          echo ">>> Detected unstaged changes. Formatting while keeping backups in .orig files."
          astyle --options=astylecfg --recursive $ASTYLEFILES > /dev/null
      fi
      echo ">>> Formatted your code. Check 'git diff' and/or *.orig files."
  else
      echo ">>> Not formatting, code does not comply with astyle rules."
      exit 1
  fi
else
    rm astyleoutput.log
    echo ">>> Code complies with astyle rules, thank you."
    exit 0
fi

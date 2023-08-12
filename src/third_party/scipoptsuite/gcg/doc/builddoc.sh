#!/bin/bash

# Generate doxygen documentation for GCG.

# Stop on error.
#set -e

# Text colors/style (bold, white, underlined, red)
W='\e[0m'
B='\e[1m'
U='\e[4m'
R='\e[91m'

# Needed for cmake compatibility
if [[ -z ${BINDIR} ]]; then export BINDIR="$PWD/../bin"; fi

# Update the version on the main page of the documentation to the one described in the
# first changelog header. Add this version and all existing versions in doc/html/
# to the docversions.html, which creates the dropdown menu
updateVersions () {( set -e
  rm -f docversions.html

  # Update version on docu main page
  sed -i -e "/@version/d" -e "/@ref RN/d" resources/main.md
  echo "@version $CURRENT_VERSION" >> resources/main.md
  echo "@ref ${CURRENT_VERSION_LINK} \"Changelog of this version\"" >> resources/main.md

  # update drop-down menu, include all folders found in html/doc-*
  # Generate dropdown menu
  for docversion in html/doc-*; do
    V=$(echo $docversion | cut -d"-" -f2)
    echo "<li><a href='../doc-${V}/index.html'>GCG ${V}</a></li>" >> docversions.html;
  done
  # sort the dropdown menu
  sort -u docversions.html -o docversions.html

  # add entry for PyGCGOpt in the dropdown
  PyGCGOpt_V=$(curl -L -s "scipopt.github.io/PyGCGOpt/CHANGELOG.html" | grep -m1 "<div class=\"brand\">PyGCGOpt" | sed -e 's/<[^>]*>//g' | sed "s/PyGCGOpt//" | tr -d " ")
  echo "<li><a href='https://scipopt.github.io/PyGCGOpt/'>PyGCGOpt (${PyGCGOpt_V})</a></li>" >> docversions.html;

  # update changelog.md (the changelog page in the documentation) automatically
  sed -i.bak "/@subpage/d" resources/misc/changelog.md && rm resources/misc/changelog.md.bak
  grep "@page RN" ../CHANGELOG | cut -d" " -f7 | sed -e "s/\.//g" -e "s/^/- @subpage RN/" >> resources/misc/changelog.md
)}

# Adds new .md pages to the table of contents in the folder
# with a file named as the folder (.md).
makeSubpageIndexing () {( set -e
  DIR=$1
  TITLE=$2
  cd $DIR
  OUT=$(basename $PWD).md
  echo "# $TITLE {#$(echo $OUT | sed 's/.md//')}" > $OUT
  # Get index list and append to .md
  ls | egrep '\.md$' | sed "/$OUT/d" | sed 's/.md//' | sed 's/^/- \@subpage /' >> $OUT

  #echo "Subpage indexing for ${DIR} built sucessfully."
  cd -  > /dev/null 2>&1
)}

# Get GCG Menu for interactive menu page
makeInteractiveMenuDocu () {( set -e
  cd resources/users/features/interactive-menu
  rm -f menu.html

  python3 getMenu.py

  cat menu_start.html.in  > menu.html
  cat menu.txt            >> menu.html
  cat menu_end.html.in    >> menu.html

  # Remove the text file that contains all menu entries (except for submenus, e.g. master/explore)
  rm menu.txt
  cd -
)}

# Check if mathjax is wanted and clone repository on a fixed working version
checkMathjax () {( set -e
  if [ "$1" == "--mathjax" ]
  then
     DOXYGEN_USE_MATHJAX="YES"
     if [ -d html/MathJax ]
     then
        printf ": updating repository\n"
        cd html/MathJax
        git stash > /dev/null 2>&1
        git checkout 2.7.7 > /dev/null 2>&1
        rm -f *.md
        cd ../..
     else
        printf ": cloning repository\n"
        cd html
        git clone https://github.com/mathjax/MathJax.git --branch=2.7.7 --single-branch --depth 1  > /dev/null 2>&1
        cd MathJax && rm -f *.md && cd ..
        cd ..
     fi
  else
    printf ": compiling without mathjax\n"
    DOXYGEN_USE_MATHJAX="NO"
  fi
)}

# Download SCIP css files, fonts etc. such that no accesses to American sites etc. are performed
getAdditionalResources () {( set -e
  mkdir -p html/bootstrap/css
  mkdir -p html/bootstrap/js
  mkdir -p html/css
  mkdir -p html/js
  mkdir -p html/img
  # Getting Bootstrap stuff
  wget https://scipopt.org/bootstrap/css/bootstrap.min.css --output-document html/bootstrap/css/bootstrap.min.css --no-check-certificate
  wget https://scipopt.org/bootstrap/css/custom.css --output-document html/bootstrap/css/custom.css --no-check-certificate
  sed -i.bak 's/https:\/\/scipopt\.org\/images/..\/..\/img/g' html/bootstrap/css/custom.css && rm html/bootstrap/css/custom.css.bak
  # Getting fonts and css
  wget https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css --output-document html/bootstrap/fonts/font-awesome.min.css
  wget https://fonts.googleapis.com/css?family=Open+Sans --output-document html/bootstrap/fonts/font-googleapis.css
  wget https://fonts.gstatic.com/s/opensans/v17/mem8YaGs126MiZpBA-UFW50bbck.woff2 --output-document html/bootstrap/fonts/font-googleapis.woff2
  wget https://fonts.gstatic.com/s/opensans/v17/mem8YaGs126MiZpBA-UFW50bbck.woff2 --output-document html/bootstrap/fonts/font-googleapis.woff2
  wget https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/fonts/fontawesome-webfont.woff2 --output-document html/bootstrap/fonts/fontawesome-webfont.woff2
  # Getting js
  wget https://scipopt.org/bootstrap/js/custom.js --output-document html/bootstrap/js/custom.js --no-check-certificate
  wget https://scipopt.org/bootstrap/js/bootstrap.min.js --output-document html/bootstrap/js/bootstrap.min.js --no-check-certificate
  wget https://code.jquery.com/jquery.min.js --output-document html/js/jquery.min.js
  # move additional resources to html folder
  cp -r resources/misc/scripts html/doc
  cp -r resources/misc/files html/doc
  mkdir -p html/doc/img/visu
  mkdir -p html/doc/bootstrap
  mkdir -p html/doc/js
  mkdir -p html/doc/css
  cp -r resources/devs/howtouse/visualizations/img/* html/doc/img/visu
  cp -r html/bootstrap html/doc
  cp -r html/js html/doc
  cp -r html/css html/doc
  cp -r html/img/newscippy.png html/doc/img/
  cp -r html/img/scribble_light_@2X.png html/doc/img/
  echo $?
)}

# Generate interactively created FAQ (txt -> php)
generateFAQ () {( set -e
  cd resources/misc/faq
  python3 parser.py --linkext shtml  && php localfaq.php > faq.inc
  cd -
)}

generateVisuArgs ()  {( set -e
  cd resources/devs/misc
  python3 -c 'import pandas as pd; pd.read_json("feature_descriptions.json").transpose().rename_axis("feature").reset_index().set_index(["groups","feature"]).sort_index().to_html("visu_args_table.html", classes="doxtable")'
  # uncomment following lines to replace dark blue and normal font with white and code font for each feature
  #awk '/<th>/{c+=1}{if(c>12){sub("<th>","<td><code>",$0)};print}' visu_args_table.html > tmp && mv tmp visu_args_table.html
  #awk '/<\/th>/{c+=1}{if(c>12){sub("</th>","</code></td>",$0)};print}' visu_args_table.html > tmp && mv tmp visu_args_table.html
  cd -
)}

generateVisuNotebookHTML () {( set -e
  cd ../stats/
  jupyter nbconvert --output-dir='../doc/resources/misc/files/' --output='visu_notebook.html' --to html visu_notebook_v*.ipynb
  cd -
  cp resources/misc/files/visu_notebook.html html/doc/files
)}

# Generate parameter file (includes SCIP params)
generateParamsFile () {( set -e
  cd ..
  "$BINDIR"/gcg -c "set default set save doc/resources/misc/parameters.set quit"
  cd -
)}

# Remove citelist.html (the Bibliography) manually from the menu (but still reachable via link)
removeBibliography () {( set -e
  cd html/doc
  sed -i.bak "/citelist/d" pages.html && rm pages.html.bak
  sed -i.bak "/citelist/d" navtreedata.js && rm navtreedata.js.bak
  sed -i.bak "s/\:\[5/\:\[4/g" navtreeindex*.js && rm navtreeindex*.js.bak # citelist is the third item in the navigation (after Users Guide and Devs Guide,
  sed -i.bak "s/\:\[6/\:\[5/g" navtreeindex*.js && rm navtreeindex*.js.bak # since Installation counts as homepage and thus 0)
  sed -i.bak "s/initResizable()\;//g" *.html
  cd -
)}

# Create Doxygen documentation for pages and source code
generateDoxy () {( set -e
  # Create index.html and gcgheader.html.
  SCIPOPTSUITEHEADER=`sed 's/\//\\\\\//g' scipoptsuiteheader.html.in | tr -d '\n'`
  DOCVERSIONS=`sed 's/\//\\\\\//g' docversions.html | tr -d '\n'`
  YEAR=`date +"%Y"`

  # Replace year by current and version by installed one
  sed -e "s/<SCIPOPTSUITEHEADER\/>/${SCIPOPTSUITEHEADER}/g" -e "s/<DOCVERSIONS\/>/${DOCVERSIONS}/g" -e "s/..\/doc/doc/g" -e "s/<YEAR\/>/${YEAR}/g" -e "s/<CURRGCG\/>/${CURRENT_VERSION}/g" < index.html.in > html/index.html
  sed -e "s/<SCIPOPTSUITEHEADER\/>/${SCIPOPTSUITEHEADER}/g" -e "s/<DOCVERSIONS\/>/${DOCVERSIONS}/g" < gcgheader.html.in > gcgheader.html

  # Set mathjax flag to export it
  DOXYGEN_USE_MATHJAX="USE_MATHJAX=${DOXYGEN_USE_MATHJAX}"

  # Build the gcg documentation.
  printf "${R}" # make doxygen errors red
  export ${DOXYGEN_USE_MATHJAX}; doxygen gcg.dxy
  printf "${W}"
)}

main () {
  # The $? conditions below are a try-catch method to alert the user
  # about the origin of the issue

  n=10 # number of steps to be performed
  i=1 # step counter

  # Find relevant documentation version.
  CURRENT_VERSION_LINK=`cat ../CHANGELOG | grep "@section" -m1 | cut -d" " -f2 | tr -d '\r'`
  CURRENT_VERSION=`cat ../CHANGELOG | grep "@section" -m1 | cut -d" " -f4 | tr -d '\r'`

  printf "${B}${U}Building GCG HTML Documentation in html/doc-${CURRENT_VERSION}${W}\n"
  mkdir -p html/doc-$CURRENT_VERSION

  # Requirement: none
  echo "[${i}/${n}] Setting the GCG version"; let "i++"
    updateVersions
    if [ $? -ne 0 ]; then printf "\n ${R}Error:${W} You modified folders or the mainpage. Please reset documentation data.\n"; fi

  # Requirement: Correctly installed git
  printf "[${i}/${n}] Checking mathjax status"; let "i++"
    checkMathjax $1
    if [ $? -ne 0 ]; then printf "\n ${R}Error:${W} Please check your internet connection and that git is installed.\n"; fi

  # Requirement: Internet connection
  echo "[${i}/${n}] Downloading additional resources"; let "i++"
    getAdditionalResources  > /dev/null 2>&1
    if [ $? -ne 0 ]; then printf " ${R}Error:${W} Please check your internet connection.\n"; fi

  # Requirement: none
  #echo "[${i}/${n}] Generating subpage indexing"; let "i++"
  #  makeSubpageIndexing "resources/devs/howtoadd/" "How to add"
  #  makeSubpageIndexing "resources/devs/howtouse/" "How to use"

  # Requirement: Correctly installed php
  echo "[${i}/${n}] Generating FAQ"; let "i++"
    generateFAQ  > /dev/null 2>&1
    if [ $? -ne 0 ]; then printf " ${R}Error:${W} Have you installed PHP and python3 correctly?\n"; fi

  # Requirement: Correctly installed GCG
  echo "[${i}/${n}] Generating GCG parameters file"; let "i++"
    generateParamsFile  > /dev/null 2>&1
    if [ $? -ne 0 ]; then printf " ${R}Error:${W} Have you installed GCG correctly?\n"; fi

  # Requirement: Correctly installed GCG
  echo "[${i}/${n}] Generating GCG interactive menu documentation"; let "i++"
    makeInteractiveMenuDocu  > /dev/null 2>&1
    if [ $? -ne 0 ]; then printf " ${R}Error:${W} Have you installed GCG and python3 correctly?\n"; fi

  # Requirement: Python3 with pandas
  echo "[${i}/${n}] Generating GCG visualization arguments documentation"; let "i++"
    generateVisuArgs  > /dev/null 2>&1
    if [ $? -ne 0 ]; then printf " ${R}Error:${W} Have you installed python3 with pandas correctly?\n"; fi

  # Requirement: Jupyter
  echo "[${i}/${n}] Generating visualization notebook HTML file"; let "i++"
    generateVisuNotebookHTML  > /dev/null 2>&1
    if [ $? -ne 0 ]; then printf " ${R}Error:${W} Have you installed Jupyter correctly?\n"; fi

  # Requirement: Doxygen, graphviz
  echo "[${i}/${n}] Generating Doxygen documentation"; let "i++"
    generateDoxy #> /dev/null 2>&1 # Show doxygen warnings!
    if [ $? -ne 0 ]; then printf " ${R}Error:${W} Have you installed Doxygen and graphviz correctly?\n"; fi

  # Requirement: none
  echo "[${i}/${n}] Finalizing"
    # remove bibliography page (used for the use case pages)
    removeBibliography  > /dev/null 2>&1
    # remove old documentation under this name
    rm -rf html/doc-${CURRENT_VERSION} gcgheader.html
    # move freshly generated docu into the desired (versionized) folder
    mv html/doc html/doc-${CURRENT_VERSION}

  printf "${B}Done!${W}\n"
}

main $@

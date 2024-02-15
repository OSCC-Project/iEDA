#!/bin/bash
SUBMODULE_FILE=".gitmodules"

IFS=$'\n'
submodule_dirs=( $(grep path $SUBMODULE_FILE | cut -d'=' -f2 | cut -d' ' -f2) )
module_names=( $(grep path $SUBMODULE_FILE | cut -d'=' -f2 | cut -d' ' -f2 | cut -d"/" -f2 | awk '{print toupper($0)}') )
git_repos=( $(grep url $SUBMODULE_FILE | cut -d'=' -f2 | cut -d' ' -f2) )

# Generate script exporting commit hashes of submodules as environment variable
echo "#!/bin/bash" > scripts/submodule_heads.sh
for i in "${!submodule_dirs[@]}"; do
  echo "export ${module_names[$i]}_HEAD=\"$(cat .git/modules/${submodule_dirs[$i]}/HEAD)\"" >> scripts/submodule_heads.sh
done

# Generate script that checkouts submodules
echo "#!/bin/bash" > scripts/checkout_submodules.sh
echo "source scripts/submodule_heads.sh" >> scripts/checkout_submodules.sh
echo "ROOT=\${PWD}" >> scripts/checkout_submodules.sh
for i in "${!submodule_dirs[@]}"; do
  echo ""  >> scripts/checkout_submodules.sh
  echo "# Initialize ${module_names[$i]}"  >> scripts/checkout_submodules.sh
  echo "[ ! \"\$(ls -A ${submodule_dirs[$i]})\" ] &&" >> scripts/checkout_submodules.sh
  echo "git clone ${git_repos[$i]} ${submodule_dirs[$i]} &&" >> scripts/checkout_submodules.sh
  echo "cd ${submodule_dirs[$i]} && git checkout \${${module_names[$i]}_HEAD} && cd \${ROOT}" >> scripts/checkout_submodules.sh
done

chmod u+x scripts/submodule_heads.sh
chmod u+x scripts/checkout_submodules.sh




find . -type f -name '*multidcp_*.py' | xargs sed -i "s/feature\['drug'/data\.feature\['drug'/g"

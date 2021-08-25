find . -type f -name '*multidcp*.sh' | xargs sed -i "s/data\/gene_expression/data\/gene_expression_for_ae\/gene_expression/g"

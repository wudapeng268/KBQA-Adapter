#!/usr/bin/env bash

bash script/test-kbqa-fold-0.sh $1 config/baseline.yml
bash script/test-kbqa-fold-0.sh $1 config/mse-all.yml
bash script/test-kbqa-fold-0.sh $1 config/mse-recon-all.yml
bash script/test-kbqa-fold-0.sh $1 config/gan-all.yml
bash script/test-kbqa-fold-0.sh $1 config/gan-recon-all.0.1.yml
bash script/test-kbqa-fold-0.sh $1 config/minus.yml

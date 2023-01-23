NV_GPU=$1 nvidia-docker run -v $HOME/src/gymnax-blines:/home/duser/gymnax-blines -it -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e WANDB_API_KEY=$(cat $HOME/.oxwhirl_wandb_api_key) gymnax:benlis /bin/bash


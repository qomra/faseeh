model=meta-llama/Llama-3.2-1B-Instruct
# share a volume with the Docker container to avoid downloading weights every run
# check if MAKNAZ_MODULES_CACHE environment variable is set to the same path otherwise set volume to $PWD/data
if [ -z "$MAKNAZ_MODULES_CACHE" ]; then
    volume=/home/jalalirs/Documents/code/arabi/maknaz/maknaz_/model
else
    volume=$MAKNAZ_MODULES_CACHE
fi

docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data   ghcr.io/huggingface/text-generation-inference:3.0.0 --model-id $model


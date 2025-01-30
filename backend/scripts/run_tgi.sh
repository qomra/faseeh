#model=/models/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08/
#model=/models/mysam/tyf-1.0-1B
model=/models/mysam/oryx-2.0-1B-Instruct
#model=/models/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/81cee02dd020268dced5fa1327e8555acce9c63c/
# share a volume with the Docker container to avoid downloading weights every run
# check if MAKNAZ_MODULES_CACHE environment variable is set to the same path otherwise set volume to $PWD/data
if [ -z "$MAKNAZ_MODULES_CACHE" ]; then
    volume=/home/jalalirs/Documents/code/arabi/maknaz/maknaz_/model
else
    volume=$MAKNAZ_MODULES_CACHE
fi

export RUST_BACKTRACE=10
docker run -e CUDA_VISIBLE_DEVICES=1  --runtime=nvidia -p 6002:6002 --gpus all --shm-size 1g  \
    -v $volume:/models \
    ghcr.io/huggingface/text-generation-inference:3.0.0 \
    --model-id $model  \
    --port 6002

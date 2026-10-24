# r2/gen.sh MANIFEST OUTPREFIX GPUS...  (cdpo1b, gs2 x4 + gs6 x4, 2 shards/GPU)
cd ${VINV_CODE:-/opt/vinvtts-submission/vinvtts}; export HF_HOME=/data/vinvtts/hf OMP_NUM_THREADS=2 LANG=C.UTF-8 LC_ALL=C.UTF-8
M=/data/vinvtts/runs/omni_dpo/cdpo1b; F=$1; O=$2; shift 2; G=("$@"); n=$((${#G[@]}*2)); i=0
for g in "${G[@]}"; do for k in 0 1; do for gs in 2 6; do
 CUDA_VISIBLE_DEVICES=$g nohup /data/vinvtts/src/omnivoice/.venv/bin/python nvfix.py $F ${O}_gs$gs $M --nbest 4 --gs $gs --shard $i/$n > ${O}_gs${gs}_$i.log 2>&1 &
done; i=$((i+1)); done; done; wait
for gs in 2 6; do cat ${O}_gs$gs.s?.jsonl > ${O}_gs$gs.jsonl; done; echo GEN_DONE

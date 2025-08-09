
. $HOME/anaconda3/etc/profile.d/conda.sh
conda activate DeepSpeed

nvidia-smi
nvidia-smi topo -m

cd /path/
python -m runs.ours_magent


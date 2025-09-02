conda create -y -n dcase2023 python==3.8.5
conda activate dcase2023
#conda install -y pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -y librosa ffmpeg sox pandas numba scipy torchmetrics youtube-dl tqdm pytorch-lightning=1.9 -c conda-forge
conda install -y  ffmpeg pandas numba scipy torchmetrics  tqdm pytorch-lightning=1.9
# conda install -y librosa  sox  youtube-dl  -c conda-forge
pip install  librosa  sox  youtube-dl


pip install tensorboard
pip install h5py
pip install thop
pip install codecarbon==2.1.4
pip install -r requirements.txt
pip install -e ../../.

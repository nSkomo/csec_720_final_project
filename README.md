# CSEC 720 Final Project — RawNet2 Anti-Spoofing

A significant portion of this code is adapted from the [ASVspoof 2021 Baseline (RawNet2)](https://github.com/asvspoof-challenge/2021/tree/main/LA/Baseline-RawNet2). Most of the modifications involve path updates and evaluation enhancements.

---

## 🔧 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/nSkomo/csec_720_final_project.git
   cd csec_720_final_project
   ```

2. Create and activate the Conda environment:
   ```bash
   conda create --name rawnet_anti_spoofing python=3.6.10
   conda activate rawnet_anti_spoofing
   ```

3. Install dependencies:
   ```bash
   conda install pytorch=1.4.0 -c pytorch
   pip install -r requirements.txt
   ```

---

## 📁 Dataset

The model is trained on the **Logical Access (LA)** subset of the [ASVspoof 2019 dataset](https://drive.google.com/drive/folders/1-4EtZZI_DJCyHEtt9jcnsvCKQ04_kmFu?usp=sharing).

---

## 🏋️‍♂️ Training

To train the model, run:

```bash
python main.py \
  --database_path="path/to/training_and_dev_sets" \
  --protocols_path="path/to/protocols" \
  --track=LA \
  --loss=CCE \
  --lr=0.0001 \
  --batch_size=32
```

---

## 🧪 Evaluation

To test your model on the **500 real** and **500 spoofed** samples from the ASVspoof 2019 LA evaluation set:

1. Replace `FLAC_DIR` in the script with your dataset directory.
2. Replace the model path with your trained or baseline model.

```bash
python evaluate.py
```

Baseline and example models are available [here](https://drive.google.com/drive/folders/1-4EtZZI_DJCyHEtt9jcnsvCKQ04_kmFu?usp=sharing).

---

## 🎵 Generating Noisy Samples

Use `music_script.py` to generate background-noise-overlaid audio samples:

1. Set `SONG_FOLDER` to a directory containing background noise audio (e.g., music, rain, etc.).
2. Set `CLIP_FOLDER` to the folder of original audio clips you want to augment.

Download the script [here](https://drive.google.com/drive/folders/1-4EtZZI_DJCyHEtt9jcnsvCKQ04_kmFu?usp=sharing).
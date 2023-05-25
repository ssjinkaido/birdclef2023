# Birdclef2023
## Training strategy
- Audio files used are .ogg. Wav files are not faster.
- Process audio using librosa, convert audio to mel-spectrogram using torchaudio.
- Slightly upsample class that has less than (5, 10, 15 samples)
- Heavy augmentation by adding multiple types of noise and modifying volume.
- Using Imagenet pre-trained weight to train on previous Birdclef datasets (2020+2021+2022).
- With these models trained on previous datasets, finetune this year's dataset (2023).
- Hyperparameters tuning using Optuna.
- Weights ensemble using SWA.
## Model used
- EfficientNetV2B1 + EfficientNetV2B2 + EfficientNetV2S, with GEM pooling, one classifier layer, high dropout rate, and drop_path_rate.
- Training time is around 55s - 75s, validation time is around 240s.
## Inference 
- Faster inference using concurrent executor in Python.
## Results 
| Models | Public LB | Private LB |
| -------- | -------- | -------- |
| 1 V2B1 + 2 V2B2 + 1 V2S | 0.82 (40/1209) | (34/1209)|

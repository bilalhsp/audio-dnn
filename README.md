# audio-dnn 🎧

This repository is dedicated to developing DNN-based models for a wide range of audio processing tasks. Unlike most models that are tailored for speech, our aim is to create a general audio processor capable of handling diverse sounds. We begin with pretraining the wav2vec2 model using data from the [Audioset](https://research.google.com/audioset/index.html) dataset, sampled at 48KHz — a higher rate than the typical 16KHz used for speech — to capture the richness of natural sounds.

We've customized the Hugging Face🤗 Trainer class to meet our pretraining requirements, adhering to the guidelines from the wav2vec2 paper. Additionally, we've utilized [audioset-utils](https://github.com/bilalhsp/audioset-utils.git) to work with Audioset metadata and selectively download categories relevant to our focus. By filtering out speech and music, we concentrate our training on natural sounds to develop a truly versatile audio processor.

We've customized the [🤗 Trainer class](https://huggingface.co/docs/transformers/en/main_classes/trainer), 
which provides functionality for training transformer-based models in a distributed fashion. 
By subclassing this trainer, we've tailored its methods to align with the pretraining approach 
specified in the [wav2vec2 paper](https://arxiv.org/abs/2006.11477).


## Table of Contents
- ⚙️ [Features](#features-⚙️)
- 🗂️ [Directory Structure](#directory-structure🗂️)
- 🔧 [Installation](#installation-🔧)
- 📄 [Configuration](#configuration-📄)
- 🚀 [Usage](#usage)
- 📜 [License](#license-📜)
- 📚 [References](#references-📚)

## Features ⚙️
- Pretraining [wav2vec2](https://arxiv.org/abs/2006.11477) model using natural sounds extracted from [Audioset](https://research.google.com/audioset/index.html) dataset.
- Customized [🤗Trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer) for pretraining.
- Utilizes 
  [audioset-utils](https://github.com/bilalhsp/audioset-utils.git) for working with 
  Audioset metadata.

## Directory Structure🗂️ 

```plaintext
audio-dnn
├── audio_dnn
│   ├── __init__.py
│   ├── data_collator.py
│   ├── data_preprocessor.py
│   ├── helpers.py
│   ├── trainer.py
│   └── utils.py
├── config
│   ├── feature_extractor_config.json
│   ├── wav2vec2_config.json
│   └── training_config.yml
├── scripts
├── slurm_scripts
├── README.md
└── setup.py
```
### Explanation of Directory Contents:

- **`audio_dnn/`** 📦: This directory contains the main implementation of the  
  package, including various modules for data processing, training,  
  and utility functions.

- **`config/`** ⚙️: This folder holds configuration files that define parameters  
  for the model architecture and training process, allowing easy adjustments.

- **`scripts/`** 📜: This directory includes example scripts for downloading,  
  preprocessing the dataset, and pretraining the wav2vec2 model.

- **`slurm_scripts/`** 🖥️: This folder contains scripts specifically designed  
  for managing job submissions in environments using the SLURM workload  
  manager.  

## Installation 🔧

To use this repository, follow these steps:

1. **Clone the Repository**
   Open your terminal and run the following command to clone the repository:

   ```bash
   git clone https://github.com/bilalhsp/audio-dnn.git
    ```
2. **Navigate to the Repository Directory**
   Change your working directory to the cloned repository:

   ```bash
   cd audio-dnn
    ```
3. **Install the Package**
   Use the following command to install the package:

   ```bash
   pip install .
   ```
   This command will install the package as a standard, non-editable installation.
4. **Modify Configuration Files (Optional)**
   If needed, you can modify the configuration files (e.g., `.json` and `.yml`) located in the `config` directory to suit your preferences.  

## Configuration 📄

The `audio-dnn` package uses configuration files to set parameters for 
training and data processing. You can find these configuration files 
in the `config` directory. The following configuration files are available:

- **`feature_extractor_config.json`**: This file contains the settings for 
  the feature extractor used in the model.
  
- **`wav2vec2_config.json`**: This file holds the model-specific parameters 
  for the wav2vec2 architecture.
  
- **`training_config.yml`**: This YAML file includes training hyperparameters, 
  such as learning rate, batch size, and number of epochs.

Feel free to modify these files to tailor the model and training 
process to your specific needs!

## Usage 🚀

To get started with the audio-dnn package, follow these steps:

1. **Download the Dataset**  
   Use the [audioset-utils](https://github.com/bilalhsp/audioset-utils.git) to download the Audioset dataset. 
   This dataset contains the natural sounds for pretraining. 
   Follow the instructions in the *audioset-utils* repository for details. 

2. **Prepare Your Dataset**  
   After downloading the dataset, use the [preprocess_dataset.py](./scripts/preprocess_dataset.py) script to ready dataset for pretraining script.
   This script utilizes [data_preprocessor.py](./audio_dnn/data_preprocessor.py) to preprocess the relevant audio data.  

3. **Pretrain the wav2vec2 Model**  
   Once your dataset is prepared, you can pretrain the wav2vec2 model using [pretrain_wav2vec2.py](./scripts/pretrain_wav2vec2.py). Following classes implement important functionality for this script;
   - [DataCollatorForPretraining](./audio_dnn/data_collator.py) is a callable object providing collation of data while making batches.
   - [CustomWav2Vec2Trainer](./audio_dnn/trainer.py) is customized trainer class, that makes training processes in line with the details described in Wav2Vec2 paper.

Now you are ready to use the audio-dnn package for your audio processing tasks!

## License 📜

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.


## References 📚

- [Wav2Vec2 Paper](https://arxiv.org/abs/2006.11477)
- [Hugging Face Trainer Documentation](https://huggingface.co/docs/transformers/en/main_classes/trainer)
- [AudioSet Dataset](https://research.google.com/audioset/)
- [AudioSet Utils GitHub Repository](https://github.com/bilalhsp/audioset-utils.git)
- [Speech-Pretraining Example Without Trainer](https://github.com/huggingface/transformers/blob/main/examples/pytorch/speech-pretraining/run_wav2vec2_pretraining_no_trainer.py)
- [Finetuning Pretrained Wav2Vec2](https://github.com/huggingface/blog/blob/main/fine-tune-wav2vec2-english.md)







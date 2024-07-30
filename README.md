
# audio-dnn

This repository is dedicated to developing DNN-based models for a wide range of audio processing tasks. Unlike most models that are tailored for speech, our aim is to create a general audio processor capable of handling diverse sounds. We begin with pretraining the wav2vec2 model using data from the [Audioset](https://research.google.com/audioset/index.html) dataset, sampled at 48KHz — a higher rate than the typical 16KHz used for speech — to capture the richness of natural sounds.

We've customized the Hugging Face Trainer class to meet our pretraining requirements, adhering to the guidelines from the wav2vec2 paper. Additionally, we've utilized [audioset-utils](https://github.com/bilalhsp/audioset-utils.git) to work with Audioset metadata and selectively download categories relevant to our focus. By filtering out speech and music, we concentrate our training on natural sounds to develop a truly versatile audio processor.

We've customized the [Hugging Face Trainer class](https://huggingface.co/docs/transformers/en/main_classes/trainer), 
which provides functionality for training transformer-based models in a distributed fashion. 
By subclassing this trainer, we've tailored its methods to align with the pretraining approach 
specified in the [wav2vec2 paper](https://arxiv.org/abs/2006.11477).


## Table of Contents
- [wav2vec2 Pretraining on Natural Sounds](#wav2vec2-pretraining-on-natural-sounds)
- [Repository Structure](#repository-structure)
- [Directory Descriptions](#directory-descriptions)
- [How to Use](#how-to-use)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## wav2vec2 Pretraining on Natural Sounds

This section details the implementation of pretraining the wav2vec2 model on natural sounds data extracted from the Audioset dataset. Speech and music labels were filtered out to focus exclusively on natural sounds. The scripts provided in this repository demonstrate how to train a general audio processor using data sampled at 48KHz.

## Repository Structure

The repository is organized into the following directories and files:

<!--
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
└── slurm_scripts
-->

<h2>Repository Structure</h2>
<p>The repository is organized into the following directories and files:</p>
<ul>
    <li>audio-dnn
        <ul>
            <li>audio_dnn
                <ul>
                    <li>__init__.py</li>
                    <li>data_collator.py</li>
                    <li>data_preprocessor.py</li>
                    <li>helpers.py</li>
                    <li>trainer.py</li>
                    <li>utils.py</li>
                </ul>
            </li>
            <li>config
                <ul>
                    <li>feature_extractor_config.json</li>
                    <li>wav2vec2_config.json</li>
                    <li>training_config.yml</li>
                </ul>
            </li>
            <li>scripts</li>
            <li>slurm_scripts</li>
        </ul>
    </li>
</ul>





"""
data_audio_preprocessor.py - Audio Preprocessing for Wav2Vec2 Model

This module contains the DataPreprocessor class, which is designed to preprocess audio data
from the audioset for use in training the Wav2Vec2 model. It includes functionality to extract
input values using a feature extractor and to filter out audio sequences that are shorter than
a specified minimum length.

Author: Bilal Ahmed
Date: 07-29-2024
License: MIT

Classes:
	DataPreprocessor: A class for preprocessing audio data for Wav2Vec2 model training.

Change Log:
- 07-29-2024: Initial version created by Bilal Ahmed.
	
"""


# import os
# from audioset_utils import get_huggingface_dataset
# from transformers import Wav2Vec2FeatureExtractor
# from datasets import Audio
import gc


class DataPreprocessor:
	"""
	Data preprocessor for the Wav2Vec2 model.

	This class takes in audio data from the audioset and
	prepares it for training the Wav2Vec2 model.
	It performs the following preprocessing steps:
	- Extracts input values from the audio samples using
	  a feature extractor.
	- Filters out audio sequences that are shorter than
	  the specified minimum length.

	Attributes:
		feature_extractor (Wav2Vec2FeatureExtractor): 
			The feature extractor used to process the audio data.
		min_length (int): The minimum length (in seconds)
			for an audio sequence to be included in training.

	Methods:
		get_input_values(batch): 
			Processes a batch of audio samples to extract input values.
		preprocess_dataset(dataset, num_proc):
			Applies preprocessing to the entire dataset.
		is_not_too_short(sample):
			Checks if an audio sequence meets the minimum length requirement.
		get_seq_indices_not_too_short(dataset):
			Identifies indices of sequences that are not too short.
		filter_short_sequences(dataset):
			Filters out sequences that do not meet the minimum length requirement.
	"""
	def __init__(self, feature_extractor, min_length=5) -> None:
		self.feature_extractor = feature_extractor
		self.min_length = min_length	# min length of a sample in seconds

	def get_input_values(self, batch):
		sample = batch['audio']
		
		batch["input_values"] = self.feature_extractor(
			sample['array'], sampling_rate=sample['sampling_rate'],
			return_tensors='np'
			).input_values[0]
		
		# saving input_length for each sequence, might not be needed for this task.
		batch["input_length"] = [batch["input_values"].shape[0]/sample['sampling_rate']]

		gc.collect()
		return batch

	def preprocess_dataset(self, dataset, num_proc=1):
		
		dataset = dataset.map(
				self.get_input_values,
				remove_columns=dataset.column_names,
				num_proc=num_proc,
				# batched=True,
			)
		return dataset

	def is_not_too_short(self, sample):
		"""Checks if dataset length is not too short"""
		return sample['input_length'][0] > self.min_length

	def get_seq_indices_not_too_short(self, dataset):
		"""Returns the list of indices of sequences that are 'good'
		meaning longer than min length."""
		good_indices = []
		all_input_lengths = dataset['input_length']
		for i in range(len(dataset)):
			if all_input_lengths[i][0] > self.min_length:
				good_indices.append(i)
		return good_indices

	def filter_short_sequences(self, dataset):
		"""Checks and filters out sequences that are shorter than
		the min. sequence length."""
		good_indices = self.get_seq_indices_not_too_short(dataset)
		dataset = dataset.select(good_indices)
		return dataset


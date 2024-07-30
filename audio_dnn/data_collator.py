"""
data_collator.py - Data Collation for Wav2Vec2 Pretraining

This module provides the DataCollatorForPretraining class, which is responsible for
preparing batches of audio data for pretraining the Wav2Vec2 model. It handles the
padding of input features, generation of masked time indices for the masked language
model objective, and sampling of negative examples for contrastive loss calculation.

Author: Bilal Ahmed
Date: 07-29-2024
License: MIT

Classes:
	DataCollatorForPretraining: A data collator class for Wav2Vec2 pretraining.

Change Log:
- 07-29-2024: Initial version created by Bilal Ahmed.
"""
import torch
from dataclasses import dataclass
from typing import Dict, List, Union
from transformers import Wav2Vec2ForPreTraining, Wav2Vec2FeatureExtractor
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices

@dataclass
class DataCollatorForPretraining:
	"""
	Data collator used for pretraining the Wav2Vec2 model.

	This class is designed to be used as a callable for collating
	batches of raw audio data into a format suitable for pretraining
	the Wav2Vec2 model. It applies padding to the input features and
	generates the necessary inputs for the model's masked language
	model (MLM) objective and contrastive task.

	Attributes:
		model (Wav2Vec2ForPreTraining): The Wav2Vec2 model being pretrained.
		feature_extractor (Wav2Vec2FeatureExtractor): The feature extractor
			for processing audio data.
		padding (Union[bool, str]): The padding strategy to apply to the
			input features. Defaults to "longest".

	Methods:
		__call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
			Processes a list of features and returns a batch for pretraining.
			Args:
				features (List[Dict[str, Union[List[int], torch.Tensor]]]): A list of features,
					where each feature is a dictionary containing input values.
			Returns:
				Dict[str, torch.Tensor]: A dictionary containing the processed batch with padded
					input values, mask time indices, and sampled negative indices.
	"""

	model: Wav2Vec2ForPreTraining
	feature_extractor: Wav2Vec2FeatureExtractor
	padding: Union[bool, str] = "longest"

	def __call__(
			self,
			features: List[Dict[str, Union[List[int], torch.Tensor]]]
		) -> Dict[str, torch.Tensor]:

		input_features = [{"input_values": feature["input_values"]} for feature in features]
		batch = self.feature_extractor.pad(
			input_features,
			padding=self.padding,
			return_tensors="pt",
		)

		device = batch['input_values'].device
		batch_size, input_seq_len = batch['input_values'].shape

		seq_len = self.model._get_feat_extract_output_lengths(input_seq_len).item()

		# to avoid computing loss on padded inputs
		if batch.get("attention_mask") is not None:
			sub_attention_mask = self.model._get_feature_vector_attention_mask(
				seq_len, batch["attention_mask"]
			)

		features_shape = (batch_size, seq_len)

		# sample randomly masked indices
		mask_time_indices = _compute_mask_indices(
			features_shape,
			self.model.config.mask_time_prob,
			self.model.config.mask_time_length,
			attention_mask=sub_attention_mask,
		)

		# sample negative indices
		sampled_negative_indices = _sample_negative_indices(
			features_shape,
			self.model.config.num_negatives,
			mask_time_indices=mask_time_indices,
		)

		batch["mask_time_indices"] = torch.tensor(mask_time_indices, dtype=torch.long, device=device)
		batch["sampled_negative_indices"] = torch.tensor(sampled_negative_indices, dtype=torch.long, device=device)

		return batch

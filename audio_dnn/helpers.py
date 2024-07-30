
import os
import yaml
import json
# import torch
from pathlib import Path
# from dataclasses import dataclass
# from typing import Dict, List, Union
from datasets import load_from_disk
from transformers import Wav2Vec2Config, Wav2Vec2ForPreTraining, Wav2Vec2FeatureExtractor
# from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices


def get_training_arguments(args):
	"""Returns an object of transformers.TrainingArguments, having
	detailed settings of the training process."""
	
	config_file_path = args.config_file_path
	with open(config_file_path) as F:
		training_config = yaml.load(F, yaml.FullLoader)

	local_output_dir = training_config.pop('local_output_dir')
	dataset_name = training_config.pop('dataset')
	model_name = training_config.pop('model_name')
	version = training_config.pop('version')
	num_of_examples_to_accumulate = training_config.pop('num_of_examples_to_accumulate')
	hf_token_path = training_config.pop('hf_token_path')
	batch_size = training_config.pop('batch_size')

	# reading hugging face login token
	with open(hf_token_path, 'r') as F:
		hf_token = F.read().strip()

	repo_id = f"{model_name}-{dataset_name}-{version}"

	# num_train_epochs = args.num_epochs
	# batch_size = args.batch_size
	total_step_to_accumulate = num_of_examples_to_accumulate/batch_size
	gradient_accumulation_steps = int(total_step_to_accumulate/args.num_tasks)
	# save_steps = 1/(num_train_epochs*2)

	training_config['output_dir'] = os.path.join(local_output_dir, repo_id)
	training_config['gradient_accumulation_steps'] = gradient_accumulation_steps
	training_config['per_device_eval_batch_size'] = batch_size
	training_config['per_device_train_batch_size'] = batch_size
	# training_config['save_steps'] = save_steps
	# training_config['eval_steps'] = save_steps
	training_config['hub_token'] = hf_token
	training_config['hub_model_id'] = f"bilalhsp/{repo_id}"
	training_config['dataloader_num_workers'] = args.num_proc

	return training_config

# Deprecated on 07-29-24
# moved to pretraining_collator.py

# @dataclass
# class DataCollatorForPretraining:

# 	model: Wav2Vec2ForPreTraining
# 	feature_extractor: Wav2Vec2FeatureExtractor
# 	padding: Union[bool, str] = "longest"

# 	def __call__(
# 			self,
# 			features: List[Dict[str, Union[List[int], torch.Tensor]]]
# 		) -> Dict[str, torch.Tensor]:

# 		input_features = [{"input_values": feature["input_values"]} for feature in features]
# 		batch = self.feature_extractor.pad(
# 			input_features,
# 			padding=self.padding,
# 			return_tensors="pt",
# 		)

# 		device = batch['input_values'].device
# 		batch_size, input_seq_len = batch['input_values'].shape

# 		seq_len = self.model._get_feat_extract_output_lengths(input_seq_len).item()

# 		# to avoid computing loss on padded inputs
# 		if batch.get("attention_mask") is not None:
# 			sub_attention_mask = self.model._get_feature_vector_attention_mask(
# 				seq_len, batch["attention_mask"]
# 			)

# 		features_shape = (batch_size, seq_len)

# 		# sample randomly masked indices
# 		mask_time_indices = _compute_mask_indices(
# 			features_shape,
# 			self.model.config.mask_time_prob,
# 			self.model.config.mask_time_length,
# 			attention_mask=sub_attention_mask,
# 		)

# 		# sample negative indices
# 		sampled_negative_indices = _sample_negative_indices(
# 			features_shape,
# 			self.model.config.num_negatives,
# 			mask_time_indices=mask_time_indices,
# 		)

# 		batch["mask_time_indices"] = torch.tensor(mask_time_indices, dtype=torch.long, device=device)
# 		batch["sampled_negative_indices"] = torch.tensor(sampled_negative_indices, dtype=torch.long, device=device)

# 		return batch

def get_wav2vec2_feature_extractor():
	# reading configurations...
	path = Path(__file__)
	FE_config_filepath = path.parents[1] / 'config/feature_extractor_config.json'
	with open(FE_config_filepath) as F:
		FE_config = json.load(F)
		
	# creating feature extractor object...
	feature_extractor = Wav2Vec2FeatureExtractor(**FE_config)
	return feature_extractor
	


def get_model_and_feature_extractor():
	"""Returns model and feature_extractor objects, using the
	configurations saved in the config dir."""
	path = Path(__file__)
	config_filepath = path.parents[1] / 'config/wav2vec2_config.json'
	with open(config_filepath) as F:
		config = json.load(F)
		
	# # creating model, adjusted for 48KHz sampling rate
	wav2vec2_config = Wav2Vec2Config(**config)
	model = Wav2Vec2ForPreTraining(wav2vec2_config)
		
	# creating feature extractor exactly the same as wav2vec2,
	# but sampling_rate=48KHz
	feature_extractor = get_wav2vec2_feature_extractor()
	return model, feature_extractor

def get_datasets():
	hf_datasets_dir = '/scratch/gilbreth/ahmedb/cache/huggingface/datasets/audioset/filtered/'
	# train_dataset = load_from_disk(os.path.join(hf_datasets_cache, "librispeech_train")) 
	split = 'test'
	test_dataset = load_from_disk(os.path.join(hf_datasets_dir, split)) 
	# split = 'training'
	# train_dataset = load_from_disk(os.path.join(hf_datasets_dir, split)) 

	train_dataset = test_dataset.select(range(1280))
	test_dataset = test_dataset.select(range(1280, 2560))

	return train_dataset, test_dataset


# Deprecated: 07-29-24
# Moved to trainer.py 

# from transformers import Trainer
# import torch

# class CustomWav2Vec2Trainer(Trainer):
# 	def __init__(self, *args, encoder_weight_decay=0.01, gradient_scale_factor=0.1, **kwargs):
# 		super().__init__(*args, **kwargs)
# 		self.encoder_weight_decay = encoder_weight_decay
# 		self.gradient_scale_factor = gradient_scale_factor

# 		# self.scaler = torch.cuda.amp.GradScaler()  # Initialize the scaler

# 	def training_step(self, model, inputs):
# 		model.train()
# 		inputs = self._prepare_inputs(inputs)

# 		# if is_sagemaker_mp_enabled():
# 		# 	loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
# 		# 	return loss_mb.reduce_mean().detach().to(self.args.device)

# 		# Forward pass with autocast for mixed precision
# 		with self.compute_loss_context_manager():
# 			loss = self.compute_loss(model, inputs)

# 			if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
# 				model = model.module
			
# 			# Apply L2 penalty to the final layer activations of the feature encoder
# 			encoder_outputs = model.wav2vec2.feature_extractor(inputs["input_values"])
# 			l2_penalty = self.encoder_weight_decay * torch.norm(encoder_outputs, p=2)
# 			loss += l2_penalty

# 		del inputs

# 		# if (
# 		# 	self.args.torch_empty_cache_steps is not None
# 		# 	and self.state.global_step % self.args.torch_empty_cache_steps == 0
# 		# ):
# 		# 	# if is_xpu_available():
# 		# 	# 	torch.xpu.empty_cache()
# 		# 	# elif is_mlu_available():
# 		# 	# 	torch.mlu.empty_cache()
# 		# 	# elif is_npu_available():
# 		# 	# 	torch.npu.empty_cache()
# 		# 	# elif is_torch_version(">=", "2.0") and is_mps_available():
# 		# 	# 	torch.mps.empty_cache()
# 		# 	if torch.cuda.is_available():
# 		# 		torch.cuda.empty_cache()

# 		kwargs = {}

# 		# # For LOMO optimizers you need to explicitly use the learning rate
# 		# if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
# 		# 	kwargs["learning_rate"] = self._get_learning_rate()

# 		if self.args.n_gpu > 1:
# 			loss = loss.mean()  # mean() to average on multi-gpu parallel training

# 		# if self.use_apex:
# 		# 	with amp.scale_loss(loss, self.optimizer) as scaled_loss:
# 		# 		scaled_loss.backward()
# 		# else:
# 		self.accelerator.backward(loss, **kwargs)

# 		if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
# 			model = model.module
# 		# Scale down gradients for the encoder
# 		for param in model.wav2vec2.feature_extractor.parameters():
# 			if param.grad is not None:
# 				param.grad.data.mul_(self.gradient_scale_factor)

# 		return loss.detach() / self.args.gradient_accumulation_steps

# 	def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
# 		# If no evaluation dataset is provided, use the default one
# 		eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

# 		# Set up evaluation
# 		self.model.eval()
# 		output = {}
		
# 		# Prepare for accumulation
# 		total_loss = 0.0
# 		total_contrastive_loss = 0.0
# 		total_diversity_loss = 0.0
# 		num_batches = 0

# 		# Create a DataLoader for evaluation
# 		dataloader = self.get_eval_dataloader(eval_dataset)

# 		# Iterate over the DataLoader
# 		for step, batch in enumerate(dataloader):
# 			# Move batch to device
# 			batch = {k: v.to(self.args.device) for k, v in batch.items()}

# 			# Forward pass
# 			with torch.no_grad():
# 				outputs = self.model(**batch)

# 			# Extract loss
# 			loss = outputs.get('loss', None)
# 			contrastive_loss = outputs.get('contrastive_loss', None)
# 			diversity_loss = outputs.get('diversity_loss', None)
# 			if loss is not None:
# 				total_loss += loss.item()
# 				total_contrastive_loss += contrastive_loss.item()
# 				total_diversity_loss += diversity_loss.item() 
# 				num_batches += 1

# 		# Compute average loss
# 		avg_loss = total_loss / num_batches if num_batches > 0 else float('nan')
# 		avg_contrastive_loss = total_contrastive_loss / num_batches if num_batches > 0 else float('nan')
# 		avg_diversity_loss = total_diversity_loss / num_batches if num_batches > 0 else float('nan')

# 		# Compute additional metrics
# 		metrics = {
# 			f"{metric_key_prefix}_loss": avg_loss,
# 			f"{metric_key_prefix}_constrast_loss": avg_contrastive_loss,
# 			f"{metric_key_prefix}_div_loss": avg_diversity_loss,
# 		}

# 		# Report metrics
# 		self.log(metrics)

# 		return metrics

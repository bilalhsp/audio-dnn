"""
trainer.py - Custom Wav2Vec2 Trainer for Pretraining

This module defines the CustomWav2Vec2Trainer class, which extends
the Hugging Face Trainer class for pretraining Wav2Vec2 models from
scratch. The trainer provides custom functionalities specific to the
Wav2Vec2 model, allowing users to train their own ASR models using
raw audio data.

Author: Bilal Ahmed
Date: 07-29-2024
License: MIT

Classes:
	CustomWav2Vec2Trainer - A custom trainer class for pretraining
		Wav2Vec2 models.

Change Log:
- 07-29-2024: Initial version created by Bilal Ahmed.
"""
import torch
from transformers import Trainer, get_polynomial_decay_schedule_with_warmup

class CustomWav2Vec2Trainer(Trainer):
	"""
	Custom trainer for pretraining Wav2Vec2 models from scratch.

	The CustomWav2Vec2Trainer class extends the Hugging Face
	Trainer class, providing custom behavior and functionalities
	for training Wav2Vec2 models on raw audio data. It is designed
	for pretraining Wav2Vec2 models without relying on any
	pre-existing weights.

	Overridden Methods:
		__init__(...): Initializes the custom trainer with additional
			arguments for encoder weight decay and gradient scaling.
		training_step(...): Executes a single training step.
		evaluate(...): Evaluates the model on the provided dataset.
	"""
	
	def __init__(
			self, *args, 
			encoder_weight_decay=10,
			gradient_scale_factor=0.1,
			gumbel_initial_temp=2,
			gumbel_min_temp=0.5,
			gumbel_temp_anealing_factor=0.999995,
			**kwargs
			):
		"""
		Initializes the custom trainer with additional arguments for
		encoder weight decay and gradient scaling.
		
		Args:
			*args: Variable length argument list for the base Trainer class.
			encoder_weight_decay (float): Weight decay factor for
				encoder parameters. Defaults to 10.
			gradient_scale_factor (float): Scale factor for gradient accumulation.
				Defaults to 0.1.
			**kwargs: Arbitrary keyword arguments for the base Trainer class.
		"""
		super().__init__(*args, **kwargs)
		self.encoder_weight_decay = encoder_weight_decay
		self.gradient_scale_factor = gradient_scale_factor
		self.gumbel_temp = gumbel_initial_temp  				# Initial temperature
		self.gumbel_min_temp = gumbel_min_temp  						# Minimum temperature
		self.gumbel_temp_anealing_factor = gumbel_temp_anealing_factor  # Annealing factor


	def compute_loss(self, model, inputs, return_outputs=False):
		"""
		Computes loss using the outputs of forward method of 
		model, customized in the following way;
			- adds L2 penalty to the activations of the final
			  layer of the feature encoder.
		"""
		outputs = model(**inputs)
		loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

		# L2 penalty on outputs of feature_encoder, mentioned in wav2vec2 paper as well.
		# Following paper provides another reference;
		# Ref. https://www.biorxiv.org/content/10.1101/2024.03.13.584776v1.full.pdf
		# Go to Method/Models pre-training for details
		loss += self.encoder_weight_decay*torch.mean(outputs['projected_states']**2)

		return (loss, outputs) if return_outputs else loss


	def training_step(self, model, inputs):
		"""
		Executes a single training step within the custom training loop.

		This method processes the inputs using the provided model and computes the loss. It is an
		essential part of the training loop where the actual forward pass, loss computation, and
		a backward pass occur. Customized for the followings;
			- scales down gradients of feature encoder layers
			- implements temperature annealing for gumbel softmax

		Args:
			model (transformers.PreTrainedModel): The model being trained. It should be a model
				instance that is compatible with the Hugging Face Transformers library.
			inputs (dict): A dictionary containing the batch of input data. The keys in the dictionary
				should correspond to the expected input format of the model, typically including input
				ids, attention masks, and labels for supervised training.

		Returns:
			torch.Tensor: The loss tensor computed during the training step. This value is used by the
				Trainer class to perform backpropagation and update the model parameters.
		"""
		model.train()
		inputs = self._prepare_inputs(inputs)

		# Forward pass with autocast for mixed precision
		with self.compute_loss_context_manager():
			# setting the gumbel softmax temperature
			if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
				model.module.set_gumbel_temperature(self.gumbel_temp)
			else:
				model.set_gumbel_temperature(self.gumbel_temp)
			loss = self.compute_loss(model, inputs)
			

			# if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
			# 	model = model.module
			
			# # Apply L2 penalty to the final layer activations of the feature encoder
			# encoder_outputs = model.wav2vec2.feature_extractor(inputs["input_values"])
			# l2_penalty = self.encoder_weight_decay * torch.norm(encoder_outputs, p=2)
			# loss += l2_penalty

		del inputs
		kwargs = {}

		if self.args.n_gpu > 1:
			loss = loss.mean()  # mean() to average on multi-gpu parallel training

		self.accelerator.backward(loss, **kwargs)

		# Ensure correct model access for multi-GPU scenarios
		if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
			model = model.module
		# Scale down gradients for the encoder
		for param in model.wav2vec2.feature_extractor.parameters():
			if param.grad is not None:
				param.grad.data.mul_(self.gradient_scale_factor)

		# Anneal the temperature
		self.gumbel_temp = max(self.gumbel_temp * self.gumbel_temp_anealing_factor, self.gumbel_min_temp)

		return loss.detach() / self.args.gradient_accumulation_steps
	
	def create_scheduler(self, num_training_steps: int, optimizer=None):
		"""Creates scheduler for learning rate, cutomized to 
		create polynomial decay schedular."""
		self.lr_scheduler = get_polynomial_decay_schedule_with_warmup(
			self.optimizer if optimizer is None else optimizer,
			num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
			num_training_steps=num_training_steps,
		)
	
		

	def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
		"""
		Evaluates the model on the provided dataset.

		This method computes evaluation metrics for the model on the specified evaluation dataset.
		It is typically used during validation or after training to assess the model's performance
		on unseen data. The evaluation metrics can include accuracy, F1-score, perplexity, etc.

		Args:
			eval_dataset (torch.utils.data.Dataset, optional): The dataset for evaluation. If not
				provided, the evaluation will be performed on the trainer's eval_dataset attribute.
			ignore_keys (List[str], optional): List of keys to ignore when computing metrics. These
				keys correspond to specific outputs from the model (e.g., auxiliary losses) that
				should not be included in the evaluation metrics.
			metric_key_prefix (str, optional): Prefix for metric keys. Defaults to "eval". For example,
				if set to "validation", the computed metrics will have keys like "validation_loss",
				"validation_accuracy", etc.

		Returns:
			Dict[str, float]: A dictionary containing evaluation metrics. The keys represent the names
				of the metrics (e.g., "eval_loss", "eval_accuracy"), and the values are the corresponding
				metric values computed on the evaluation dataset.
		"""
		# If no evaluation dataset is provided, use the default one
		eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

		# Set up evaluation
		self.model.eval()
		output = {}
		
		# Prepare for accumulation
		total_loss = 0.0
		total_contrastive_loss = 0.0
		total_diversity_loss = 0.0
		num_batches = 0

		# Create a DataLoader for evaluation
		dataloader = self.get_eval_dataloader(eval_dataset)

		# Iterate over the DataLoader
		for step, batch in enumerate(dataloader):
			# Move batch to device
			batch = {k: v.to(self.args.device) for k, v in batch.items()}

			# Forward pass
			with torch.no_grad():
				outputs = self.model(**batch)

			# Extract loss
			loss = outputs.get('loss', None)
			contrastive_loss = outputs.get('contrastive_loss', None)
			diversity_loss = outputs.get('diversity_loss', None)
			if loss is not None:
				total_loss += loss.item()
				total_contrastive_loss += contrastive_loss.item()
				total_diversity_loss += diversity_loss.item() 
				num_batches += 1

		# Compute average loss
		avg_loss = total_loss / num_batches if num_batches > 0 else float('nan')
		avg_contrastive_loss = total_contrastive_loss / num_batches if num_batches > 0 else float('nan')
		avg_diversity_loss = total_diversity_loss / num_batches if num_batches > 0 else float('nan')

		# Compute additional metrics
		metrics = {
			f"{metric_key_prefix}_loss": avg_loss,
			f"{metric_key_prefix}_constrast_loss": avg_contrastive_loss,
			f"{metric_key_prefix}_div_loss": avg_diversity_loss,
		}

		# Report metrics
		self.log(metrics)

		return metrics
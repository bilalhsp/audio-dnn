import os
import time
import argparse
import torch.distributed as dist
from huggingface_hub import login
from transformers import TrainingArguments

#local
from audio_dnn import DataCollatorForPretraining, CustomWav2Vec2Trainer
from audio_dnn import get_model_and_feature_extractor, get_datasets, get_training_arguments
from audio_dnn import utils


def init_distributed_mode(args):
	"""Initialize process groups distributed training"""
	if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
		args.rank = int(os.environ['RANK'])
		args.world_size = int(os.environ['WORLD_SIZE'])
		args.distributed = True
		dist.init_process_group(backend='nccl', init_method='env://')
		dist.barrier()
	else:
		print("Not running in distributed mode")
		args.distributed = False
		
def train_wav2vec2(args):
	
	file_path = "/home/ahmedb/projects/Wav2Letter/audio-processing/config/training_config.yml"
	args.config_file_path = file_path
	
	# with open(file_path) as F:
	# 	training_config = yaml.load(F, yaml.FullLoader)

	# local_output_dir = training_config.pop('local_output_dir')
	# dataset_name = training_config.pop('dataset')
	# model_name = training_config['model_name']
	# version = training_config.pop('version')
	# num_of_examples_to_accumulate = training_config.pop('num_of_examples_to_accumulate')
	
	# hf_token_path = training_config.pop('hf_token_path')

	# with open(hf_token_path, 'r') as F:
	# 	hf_token = F.read().strip()

	
	
	# repo_id = f"{model_name}-{dataset_name}-{version}"

	# num_train_epochs = args.num_epochs
	# batch_size = args.batch_size
	# total_step_to_accumulate = num_of_examples_to_accumulate/batch_size
	# gradient_accumulation_steps = int(total_step_to_accumulate/args.num_tasks)
	# save_steps = 1/(num_train_epochs*2)

	# training_config['output_dir'] = os.path.join(local_output_dir, repo_id)
	# training_config['gradient_accumulation_steps'] = gradient_accumulation_steps
	# training_config['per_device_eval_batch_size'] = batch_size
	# training_config['per_device_train_batch_size'] = batch_size
	# training_config['num_train_epochs'] = num_train_epochs
	# training_config['save_steps'] = save_steps
	# training_config['eval_steps'] = save_steps
	# training_config['hub_token'] = hf_token
	# training_config['hub_model_id'] = f"bilalhsp/{repo_id}"
	# training_config['dataloader_num_workers'] = args.num_proc


	training_config = get_training_arguments(args)
	
	login(training_config['hub_token'])
	print(f"Login to hub successful...!")

	
	training_args = TrainingArguments(**training_config)
	
	# get model and feature extractor objects...
	model, feature_extractor = get_model_and_feature_extractor()
	
	# datasets and data collator objects...
	train_dataset, test_dataset = get_datasets()
	data_collator = DataCollatorForPretraining(model=model, feature_extractor=feature_extractor)
	
	# initialize trainer
	trainer = CustomWav2Vec2Trainer(
		model=model,
		data_collator=data_collator,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=test_dataset,
		tokenizer=feature_extractor,
		encoder_weight_decay=10,  # L2 regularization penalty
	    gradient_scale_factor=0.1,  # Scale down the gradients by a factor of 10
		gumbel_initial_temp=2,
        gumbel_min_temp=0.5,
        gumbel_temp_anealing_factor=0.999995,
	)
	
	print(f"Starting training...!")

	if args.resume_from_checkpoint:
		checkpoint_dir = utils.get_latest_checkpoint_dir(training_config['output_dir'])
		trainer.train(resume_from_checkpoint=checkpoint_dir)
	else:
		trainer.train()
	trainer.push_to_hub()	
	print(f"Done training...!")

	
	# file_path = "/home/ahmedb/projects/Wav2Letter/audio-processing/config/training_config.yml"

	# with open(file_path) as F:
	# 	config = yaml.load(F, yaml.FullLoader)

	# # os.environ['HUGGINGFACE_HUB_TOKEN'] = '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'

	# # login(os.environ['HUGGINGFACE_HUB_TOKEN'])
	# # print(f"Login to hub successful...!")



	# model_name = "wav2vec2-48KHz"
	# dataset_name = 'audioset-natual-sounds'
	# if args.version != 0:
	# 	dataset_name += f'-v{args.version}'
	# repo_id = f"{model_name}-{dataset_name}"
	# repo_path = f'/scratch/gilbreth/ahmedb/cache/huggingface/models/{repo_id}'
	# # repo = Repository(
	# # 	repo_path,
	# # 	clone_from=f"bilalhsp/{repo_id}",
	# # 	token=True,
	# # 	git_user="bilalhsp",
	# # 	git_email="bilalhsp@gmail.com",
	# # 	# add_to_git_credential=True,
	# # 	)
	
	# # print(f"Repo created successfully...")

	# # sampling_rate = 48000
	# num_of_examples_to_accumulate = 640	# nearly 1.8 hours of data
	# batch_size = args.batch_size
	# total_step_to_accumulate = num_of_examples_to_accumulate/batch_size
	# gradient_accumulation_steps = int(total_step_to_accumulate/args.num_tasks)
	# num_train_epochs = args.num_epochs
	# print(f"gradient accumulation steps: {gradient_accumulation_steps}")

	# save_steps = 1/(num_train_epochs*2)
	# print(f"saving every {save_steps}th step...")

	# model, feature_extractor = get_model_and_feature_extractor()

	# train_dataset, test_dataset = get_datasets()
	# data_collator = DataCollatorForPretraining(model=model, feature_extractor=feature_extractor)

	# training_args = TrainingArguments(
	# 	output_dir=repo_path,
	# 	gradient_checkpointing=False, 
	# 	group_by_length=False,
	# 	gradient_accumulation_steps=gradient_accumulation_steps,
	# 	per_device_eval_batch_size=batch_size,
	# 	num_train_epochs=num_train_epochs,
	# 	per_device_train_batch_size=batch_size,
		
	# 	# logging...
	# 	logging_strategy='steps',
	# 	logging_steps=1000,

	# 	# save and eval strategy...
	# 	save_strategy='steps',
	# 	save_steps=save_steps,
	# 	save_total_limit=2,
	# 	eval_strategy='steps',
	# 	eval_steps=save_steps,

	# 	learning_rate=1e-4,
	# 	weight_decay=0.005,
	# 	warmup_ratio=0.1,
		
	# 	fp16=True,
	# 	report_to=["tensorboard"],
	# 	load_best_model_at_end=True,
	# 	metric_for_best_model="loss",
	# 	# prediction_loss_only=True,
	# 	greater_is_better=False,
	# 	push_to_hub=True,
		
	# 	hub_token=os.environ['HUGGINGFACE_HUB_TOKEN'],
	# 	hub_model_id=f'bilalhsp/{repo_id}',
		
	# 	ddp_find_unused_parameters=True,  # Add this for DDP
	# 	ddp_backend='nccl',
		

	# 	dataloader_num_workers=args.num_proc,
	# 	)
	

	
	# trainer = CustomTrainer(
	# 	model=model,
	# 	data_collator=data_collator,
	# 	args=training_args,
	# 	train_dataset=train_dataset,
	# 	eval_dataset=test_dataset,
	# 	tokenizer=feature_extractor,
	# 	# compute_metrics=compute_eval_metrics,
	# )

	# # # Run evaluation
	# # print(f"computing evaluation...!")
	# # metrics = trainer.evaluate()
	# # print(f"Here is the output of evaluate: {metrics}")
	# print(f"Starting training...!")
	# trainer.train()
	# trainer.push_to_hub()	
	# print(f"Done training...!")
	

# ------------------  get parser ----------------------#

def get_parser():
	parser = argparse.ArgumentParser(
		description='This script is used to pretrain wav2vec2 model on ' +
		'unlabelled audio dataset, using distributed training over multiple GPUs.',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
	parser.add_argument(
		'-n','--num_proc', dest='num_proc', type=int, default=1,
		help="Specify the number of processes per task"
	)
	parser.add_argument(
		'-t','--tasks', dest='num_tasks', type=int, default=1,
		help="Specify the total number of tasks."
	)
	parser.add_argument(
		'-r','--resume_from_checkpoint', dest='resume_from_checkpoint',
		action='store_true', default=False,
		help="Specify if want to resume from latest saved checkpoint."
	)

	return parser

# ------------------  main function ----------------------#

if __name__ == "__main__":
	start_time = time.time()
	parser = get_parser()
	args = parser.parse_args()
	print(f"Process rank: {os.environ['RANK']}")
	print(f"Local rank: {int(os.environ['LOCAL_RANK'])}")
	print(f"World size: {os.environ['WORLD_SIZE']}")

	# display the arguments passed
	for arg in vars(args):
		print(f"{arg:15} : {getattr(args, arg)}")

	# dist.init_process_group(backend="gloo|nccl")
	init_distributed_mode(args)
	train_wav2vec2(args)
	elapsed_time = time.time() - start_time
	print(f"It took {elapsed_time/60:.1f} min. to run.")
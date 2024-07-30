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
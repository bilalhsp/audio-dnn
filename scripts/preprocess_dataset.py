import os
import time
import argparse

from datasets import Audio, load_from_disk
from transformers import Wav2Vec2FeatureExtractor

from audioset_utils import get_huggingface_dataset
from audio_dnn import DataPreprocessor
from audio_dnn import get_wav2vec2_feature_extractor


def preprocess_audioset(args):

	audioset_dir = '/scratch/gilbreth/ahmedb/cache/huggingface/datasets/audioset'

	# sampling_rate = 48000
	# feature_extractor = Wav2Vec2FeatureExtractor(
	# 	do_normalize=True,
	# 	feature_size=1,
	# 	padding_side="right",
	# 	padding_value=0.0,
	# 	return_attention_mask=True,
	# 	sampling_rate=sampling_rate,
	# 	)
	feature_extractor = get_wav2vec2_feature_extractor()
	sampling_rate = feature_extractor.sampling_rate
	processor = DataPreprocessor(feature_extractor)
	
	num_proc = args.num_proc
	if args.training:
		split = 'training'
	else:
		split = 'test'
	

	if args.preprocess:
		metadata_file = 'audioset_metadata.csv'
		dataset = get_huggingface_dataset(os.path.join(audioset_dir, split), metadata_file)
		dataset = dataset.cast_column('audio', Audio(sampling_rate=sampling_rate))
		dataset = processor.preprocess_dataset(dataset, num_proc=num_proc)

	else:
		processed_dir = os.path.join(audioset_dir, 'processed', split)
		dataset = load_from_disk(processed_dir) 

	print(f"Filtering out sequences shorter than {processor.min_length} sec.")
	dataset = processor.filter_short_sequences(dataset)
	processed_dir = os.path.join(audioset_dir, 'filtered', split)
	# os.makedirs(processed_dir)
	dataset.save_to_disk(processed_dir, num_proc=num_proc)

	print(f"Dataset processed and saved to disk.")



# ------------------  get parser ----------------------#

def get_parser():
	parser = argparse.ArgumentParser(
		description='This is to filter and download Audioset dataset',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
	parser.add_argument(
		'-tr','--training', dest='training', action='store_true', default=False,
		help="Specify if training data to be downloaded."
	)
	parser.add_argument(
		'-n','--num_proc', dest='num_proc', type=int, default=1,
		help="Specify the number of processes."
	)
	parser.add_argument(
		'-p','--preprocess', dest='preprocess', action='store_true', default=True,
		help="Specify if preprocessing needs to be done."
	)

	return parser


# ------------------  main function ----------------------#

if __name__ == '__main__':

	start_time = time.time()
	parser = get_parser()
	args = parser.parse_args()

	# display the arguments passed
	for arg in vars(args):
		print(f"{arg:15} : {getattr(args, arg)}")

	preprocess_audioset(args)
	elapsed_time = time.time() - start_time
	print(f"It took {elapsed_time/60:.1f} min. to run.")
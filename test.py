import ujson as json
from util import batch
import random

train_dataset =json.load(open("Training_data_wa.json", 'r'))
random.shuffle(train_dataset)

batch_size = 1
for single_batch in batch(train_dataset, batch_size):
	for question in single_batch:
		print("HEY")
		for para in question:
			print("we found para!")
			for sentence in para[2]:
				print("we found sentence!")
				print(len(para[0]))

	break

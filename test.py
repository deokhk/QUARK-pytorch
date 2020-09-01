import ujson as json

dev_data = json.load(open("Dev_data_for_qa.json"))
gold_data = json.load(open("hotpot_dev_distractor_v1.json"))
for gold_d in gold_data:
	print(gold_d)
	break
 
for data in dev_data:
	print(data)
	break 

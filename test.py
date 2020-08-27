import ujson as json

dev_data = json.load(open("hotpot_dev_distractor_v1.json"))
predicted_data = json.load(open("dev_distractor_pred.json"))

for idx, gold_data in enumerate(dev_data):
    answer = gold_data['answer']
    sf = gold_data['supporting_facts']
    id = gold_data['_id']
    _type = gold_data['type']
    if _type == "comparison":
        predicted_answer = predicted_data['answer'][id]
        predicted_sf = predicted_data['sp'][id]

        print("============================")
        print("answer: ", answer)
        print("predicted_answer: ", predicted_answer)
        print("supporting facts: ", sf)
        print("predicted sf: ", predicted_sf)
        print("============================")    
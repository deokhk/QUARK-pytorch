from run import predict
from hotpot_evaluate_v1 import eval
print("Prediction started..")
predict("hotpot_dev_distractor_v1.json", "dev_distractor_pred.json")
print("Prediction done!")
eval("dev_distractor_pred.json", "hotpot_dev_distractor_v1.json")
print("Evaluation done!")






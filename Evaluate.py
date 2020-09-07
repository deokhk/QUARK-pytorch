from run import predict
from hotpot_evaluate_v1 import eval
from QA_module import train_and_evaluate_QA_module

print("Now training QA module based on SpanBERT..")
train_and_evaluate_QA_module()
print("Prediction started..")
predict("hotpot_dev_distractor_v1.json", "dev_distractor_pred.json")
print("Prediction done!")
eval("dev_distractor_pred.json", "hotpot_dev_distractor_v1.json")
print("Evaluation done!")







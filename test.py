import torch 
import os
def add_test(a, b):
    print("a+b is:{}".format(a+b))

add_test(5,3)

if torch.cuda.is_available():
    print("cuda is availabe!")
else:
    print("cuda is not available!")
from multiprocessing.sharedctypes import Value
import os
# from statistics import LinearRegression
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)
# ----------------------------------------------------------
import torch
from torch import nn
import math
# from utils.logis_database import FeelingClassify
# from torch.utils.data import DataLoader

class LogisticRegression(nn.Module):
    def __init__(self, num_feature, num_class, learning_rate=0.001, momentum=0.9, weight_decay=0.01):
        # Initial setup for model
        super(LogisticRegression, self).__init__()
        self.num_feature = num_feature
        self.num_class = num_class
        self.learning_rate = learning_rate
        self.momentum = momentum 
        self.weight_decay = weight_decay
        

        # Model parameter using for prediction
        self.params = torch.ones(num_class, num_feature) # Must transpose before make prediction
        # self.bias = torch.zeros(1, num_class).to
        self.params_change = torch.zeros(num_class, num_feature)

        # Model optimization
        self.has_change = False
        self.grad = torch.ones(num_class, num_feature)
    
    def CEloss(self, output_probability, true_output):
        # true output size: 3750, 6
        # output probability: 3750, 6
        for class_index, params_class in enumerate(self.grad):
            prob = output_probability[:,class_index].item()
            label = true_output[:, class_index].item()
            # print('prob-label', prob, label)
            loss_grad = (label*-1.0/(prob*math.log(2))) + (1-label)*1.0/((1-prob)*math.log(2))
            self.grad[class_index].mul_(loss_grad) 
            # print('CEloss gradient', self.grad)

        total_loss = 0.0
        for sample_probability, sample_out in zip(output_probability, true_output):
            for class_probability, class_sample in zip(sample_probability, sample_out):
                prob = class_probability.item()
                true_label = class_sample.item()
                total_loss += -true_label*math.log2(prob)-(1-true_label)*math.log2(1-prob)
        
        for parameter_row in self.params:
            for params in parameter_row:
                total_loss += self.weight_decay*(pow(params.item(), 2)) 

        return total_loss

    def __sigmoid_calc(self, number):
        return  1.0/(1 + math.exp(-number))
     
    def sigmoid(self, linear_predicted):
        # Calculate gradient descent when using sigmoid function
        for class_index, params_class in enumerate(self.grad):
            class_linear = linear_predicted[:,class_index].item()
            # print('Grad sigmoid: ',class_index, class_linear)
            sigmoid_grad = self.__sigmoid_calc(class_linear)*(1-self.__sigmoid_calc(class_linear))
            self.grad[class_index].mul_(sigmoid_grad) 
            # print('Sigmoid grad: ', self.grad)
        
        # print('Sigmoid grad: ', self.grad)
        label_probability = torch.empty(1, self.num_class)
        label_probability.copy_(linear_predicted)

        for class_index in range(self.num_class):
            # print('Linear Predict',linear_predicted[:, class_index])
            # print(class_index)
            label_probability[:, class_index] = self.__sigmoid_calc(linear_predicted[:, class_index].item())  
        # quit(0) 
        return label_probability
            
    def linear(self, input_data):
        # Calculate gradient descent when make a linear progress 
        self.grad = torch.mm(torch.transpose(input_data, 0, 1), torch.ones(1, self.num_class))
        self.grad = torch.transpose(self.grad, 0, 1)
        # print('Linear grad:', self.grad)
        predicted_linear = torch.mm(input_data, torch.transpose(self.params, 0, 1))
        # print('Predicted Linear,', predicted_linear)
        return predicted_linear

    def predict_class(self, class_probability):
        class_tensor = torch.zeros_like(class_probability)
        value, indice = torch.max(class_probability, 1)
        
        class_tensor[:, indice] = 1
        return class_tensor
    
    def load_weight(self, params):
        self.params.copy_(params)

    def parameters(self):
        return self.params

    def __call__(self, input_data):
        linear_prediction = self.linear(input_data)
        return self.sigmoid(linear_prediction)
        
    def zero_grad(self):
        self.grad = torch.zeros(self.num_class, self.num_feature)

    def train(self):
        self.grad = self.grad + self.weight_decay*self.params

        if self.has_change == False:
            self.params_change.copy_(self.grad)
        else:
            self.params_change = self.momentum*self.params_change + (1 - self.momentum)*self.grad

        self.params = self.params - self.learning_rate*self.params_change


# dataset = FeelingClassify()
# data_loader = DataLoader(dataset=dataset)
# model = LogisticRegression(3, 6)
# epochs = 20
# for epoch in range(epochs):
#     total_loss = 0 
#     total_samples = 0
#     for input, output in data_loader:
#         # print('Input: ',input)
#         # print('Output:', output)
#         # print(model(input))
#         predicted = model(input)
#         total_loss += model.CEloss(predicted, output)
#         total_samples += 1
#         model.train()
#         model.zero_grad()

#         # quit(0)    

#     print(f'Epoch {epoch} with mean loss: {total_loss*1.0/total_samples}')

# true_samples = 0
# total_samples = 0
# for input, output in data_loader:
#     predicted_prob = model(input)
#     predicted_class = model.predict_class(predicted_prob)

#     if torch.all(output.eq(predicted_class)).item() == True:
#         true_samples += 1

#     total_samples += 1

# print('Probability of logis regression: ', true_samples*1.0/total_samples)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = LogisticRegression(11, 6).to(device)

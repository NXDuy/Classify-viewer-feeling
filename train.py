from numpy import zeros
from sklearn.metrics import f1_score
import torch
from logis_model.model import LogisticRegression
from utils.logis_database import FeelingClassify
from os.path import exists
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import math

def get_args(n_features):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    training_params ={
        'learning_rate': 0.001,
        'batch_size': 50,
        'epochs': 20,
        'device': device,
        'momentum': 0.9, 
        'n_features': n_features,
        'n_samples': 0.7 
    }

    testing_params = {
        'batch_size': 40,
        'n_samples': 0.3,
        'device': device
    }
    return training_params, testing_params

def load_model(training_params):
    input_dim = training_params['n_features']
    learning_rate = training_params['learning_rate']
    # batch_size = training_params['batch_size']
    momentum = training_params['momentum']
    device = training_params['device']
    # MAX_LOSS = 1e5

    model = LogisticRegression(
                num_feature=input_dim, num_class=6, learning_rate=learning_rate, 
                momentum=momentum, weight_decay=0.001)

    best_params = torch.ones_like(model.parameters())
    best_params.copy_(model.parameters())
    best_f1_score = 0

    if exists("checkpoint/logis.pth"):
        last_model = torch.load('checkpoint/logis.pth', map_location=device)
        model.load_weight(last_model['params'])
        best_f1_score = last_model['f1_score']
        best_params = last_model['params']
    # print('ahihi')
    return model, best_params, best_f1_score

def save_model(best_params, best_f1_score):
    params = {
        'params': best_params,
        'f1_score': best_f1_score 
    }
    torch.save(params, "checkpoint/logis.pth")

def split_train_test():
    database = FeelingClassify()
    n_samples = database.n_samples
    training_params, testing_params = get_args(database.n_features)

    train_size = int(n_samples*training_params['n_samples'])
    test_size = n_samples - train_size
    # print('Train_size & test size:', train_size, test_size)

    train_set, test_set = random_split(database, [train_size, test_size])
    train_loader = DataLoader(train_set)
    test_loader = DataLoader(test_set)

    return training_params, train_loader, test_loader

def train_and_evaluate():
    training_params ,train_loader, test_loader = split_train_test()
    # load_model(training_params)
    epoch_error = train(train_loader, training_params)
    evaluate(training_params, test_loader)
    plot_loss(epoch_error)

def plot_loss(epoch_loss):
    epoch_x_axis = [value[0] for value in epoch_loss]
    loss_y_axis = [value[1] for value in epoch_loss]

    plt.plot(epoch_x_axis, loss_y_axis)
    plt.xlabel('Epoch')
    plt.ylabel('Error of model')
    plt.show()
    return

def precision_recall_calculate(false_predicted, true_predicted, output_class):
    num_class = output_class.shape[1]
    precision = 0
    recall = 0
    f1_score = 0

    for class_index in range(num_class): 
        true_positive = true_predicted[:, class_index].item()
        false_positive = false_predicted[:, class_index].item()
        true_class = output_class[:, class_index].item()

        cur_class_precision = true_positive/(true_positive + false_positive + 1e-8)
        cur_class_recall = true_positive/(true_class + 1e-8)
       
        f1_score += 2*cur_class_precision*cur_class_recall/(cur_class_precision+cur_class_recall+1e-8)
        precision += cur_class_precision       
        recall += cur_class_recall
        
    return precision/num_class, recall/num_class, f1_score/num_class


def train(train_loader, training_params):
    
    epochs = training_params['epochs'] 
    device = training_params['device']

    model, best_params, best_f1_score = load_model(training_params=training_params)
    epoch_error =  list()
    # print(model.parameters(), best_params, best_loss)
    for epoch in range(epochs):
        total_loss = 0
        total_samples = 0

        total_false_predicted = torch.zeros(1, 6)
        total_output_class = torch.zeros(1, 6)
        total_true_predicted = torch.zeros(1, 6)

        # model.learning_rate = lr_schedular(cur_epoch=epoch+1, lr=model.learning_rate, lr_decay=0.01, epoch_decay=250)
        for input, output in train_loader:
            predicted = model(input)
            total_loss += model.CEloss(predicted, output)
            total_samples += 1
            model.train()
            model.zero_grad()

            predicted_class = model.predict_class(predicted)
            if torch.all(output.eq(predicted_class)).item() == True:
                total_true_predicted += predicted_class
            else:
                total_false_predicted += predicted_class

            # total_predicted_class = total_predicted_class + predicted_class
            total_output_class = total_output_class + output
            # sum_class = sum_class + output
            # print(output)

        epoch_precision, epoch_recall, epoch_f1_score = precision_recall_calculate(total_false_predicted, total_true_predicted, total_output_class)

        if math.isnan(total_loss) == False and best_f1_score < epoch_f1_score:
            best_f1_score = epoch_f1_score
            best_params.copy_(model.parameters())
        
        epoch_error.append([epoch, total_loss*1.0/total_samples])

        print(f'{epoch} with loss mean {total_loss*1.0/total_samples}')
        print(f'{epoch} with precision {epoch_precision}')
        print(f'{epoch} with recall {epoch_recall}')
        print('-------------------------------------------------')
        # print('Total class:', sum_class)

    save_model(best_params=best_params, best_f1_score=best_f1_score)
    return epoch_error

def evaluate(training_params, test_loader):
    model = load_model(training_params)
    device = training_params['device']

    model, best_params, best_f1_score = load_model(training_params=training_params)
    # print('Eval: ',model.parameters(), best_params, best_loss)
    # sum_class = torch.zeros(1, 6)
    true_samples = 0
    total_samples = 0

    total_false_predicted = torch.zeros(1, 6)
    total_output_class = torch.zeros(1, 6)
    total_true_predicted = torch.zeros(1, 6)

    for input, output in test_loader:
        predicted_prob = model(input)
        predicted_class = model.predict_class(predicted_prob)

        if torch.all(output.eq(predicted_class)).item() == True:
            true_samples += 1
            total_true_predicted += predicted_class
        else:
            total_false_predicted += predicted_class

        total_output_class += output
        total_samples += 1
        
    
    eval_precision, eval_recall, eval_f1_score = precision_recall_calculate(total_false_predicted, total_true_predicted, total_output_class)
    print(f'Mean Loss For testing data: {true_samples*1.0/total_samples}')
    print(f'Precision For testing data: {eval_precision}')
    print(f'Recall For testing data: {eval_recall}')

if __name__ == '__main__':
    train_and_evaluate()
    
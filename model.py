import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x): #Forward propogation
        x = F.relu(self.linear1(x)) # relu activation function for first layer to cater error possibilities and solves gradiant decent problems
        x = self.linear2(x) # for second layer
        return x

    def save(self, file_name='model.pth'): # pytorch file to save the model
        model_folder_path = './model'
        if not os.path.exists(model_folder_path): #if not exists
            os.makedirs(model_folder_path) #create file

        file_name = os.path.join(model_folder_path, file_name) #join file path and file name
        torch.save(self.state_dict(), file_name) # save file thorugh pytorch in state_dictionary

class QTrainer:
    def __init__(self, model, lr, gamma): 
        self.lr = lr
        self.gamma = gamma # (0-1) how much value to be given to updated rewards
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr) #Adam optimizer is used to update network weights
        self.criterion = nn.MSELoss() # loss funtion of Mean Squared Error

    def train_step(self, state, action, reward, next_state, done): # parameters from train_memory in agent.py
        
        state = torch.tensor(state, dtype=torch.float)  # (Parameter, datatype)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        
        if len(state.shape) == 1:       # One dimensional torch tensor
            state = torch.unsqueeze(state, 0) # unsqueeze function returns a new tensor with a dimension of size one
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # Predicting Q values with current state

        # Update Rule Simplified (From Bellman Ford Equation)
        # Part 1: Q = model.predict(state_0)
        # Part 2: Qnew = Reward + gamma . max( Q(state_1) )

        pred = self.model(state) # prediction for any action taken
        target = pred.clone()  # clone for new q value for every action
        
        for idx in range(len(done)): # if reward is achieved
            Q_new = reward[idx]
            if not done[idx]: # if reward is not aquired
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new # argmax gives index number having greatest value 

        
        self.optimizer.zero_grad()  #zero_grad empty the gradiants
        loss = self.criterion(target, pred)  # Calculating loss funtion b/w target and prediction
        loss.backward()  # Applying back propagation

        self.optimizer.step() # Update New Gradiants
from torch.nn import Module
import torch.nn as nn

def solution_data():
    pass

#64x64 px images
class model(Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(1,3,(3,3))
        #3x 62 x 62
        self.max1 =  nn.MaxPool2d(2,2)
        #3 x 31 x 31
        self.conv2 = nn.Conv2d(3,16,(2,2))
        #16 x 30 x 30
        self.max2 =  nn.MaxPool2d(2,2)
        self.liner = nn.Linear(15 * 15 * 16, 120)
        self.liner2 = nn.Linear(120, 84)
        self.liner3 = nn.Linear(84, 10)
    
    def forward(self,x):
        x = nn.functional.relu(self.conv1, x)
        x = nn.functional.relu(self.conv2, x)
        x = nn.functional.relu(self.liner1, x)
        x = nn.functional.relu(self.liner2, x)
        x = nn.functional.relu(self.liner3, x)
        return x
    
    


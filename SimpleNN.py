class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        
        self.layer1 = nn.Linear(2, 8)   # entrada 2 features -> 8 neuronas
        self.layer2 = nn.Linear(8, 4)   # 8 -> 4
        self.layer3 = nn.Linear(4, 1)   # salida 1

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x

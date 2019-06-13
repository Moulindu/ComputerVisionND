import torch
import torch.nn as nn
import torchvision.models as models




class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        
        super(DecoderRNN, self).__init__()
        
        #self.hidden_dim = (torch.zeros(num_layers, 1, hidden_size).cuda(), torch.zeros(num_layers, 1, hidden_size).cuda())
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        pass
    

    
    def forward(self, features, captions):
        
        x = self.embed(captions[:,:-1])
        #print(x.shape)
        x = torch.cat((features.unsqueeze(1), x), 1)
        
        #print(x.shape)
        
        hiddens, _ = self.lstm(x)
        
        output = self.fc(hiddens)
        
        return output
        
        
        
        
        
        pass

    def sample(self, inputs, state=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        outputs = []
        
        
        
        for i in range(max_len):
            
            hidden, state = self.lstm(inputs, state)
            
            #print("hidden:", hidden, "state:", state)
            
            output = self.fc(hidden.squeeze(1))
            
            #print("output:",output) #"and max value:", output.squeeze(0).max())
            
            #print("output shape:", output.shape)
            
            _, index = output.max(1)
            
            #print("index:",index)
            #print("value:", _)
            
            #print("index shape:", index.shape)
            
            outputs.append(index.item())
            
            inputs = self.embed(index).unsqueeze(1)
            
            #print("new input's shape:", inputs.shape)
            
        #outputs = torch.stack(outputs, 1)
            
        return outputs
            
            
        pass
            #print("output shape:", output.shape)
            
            _, index = output.max(1)
            
            #print("index:",index)
            #print("value:", _)
            
            #print("index shape:", index.shape)
            
            outputs.append(index.item())
            
            inputs = self.embed(index).unsqueeze(1)
            
            #print("new input's shape:", inputs.shape)
            
        #outputs = torch.stack(outputs, 1)
            
        return outputs
            
            
        pass

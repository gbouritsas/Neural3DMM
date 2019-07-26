import torch
import copy
from tqdm import tqdm
import numpy as np


def test_autoencoder_dataloader(device, model, dataloader_test, shapedata, mm_constant = 1000):
    model.eval()
    loss = 0
    l1_loss = 0
    l2_loss = 0
    shapedata_mean = torch.Tensor(shapedata.mean).to(device)
    shapedata_std = torch.Tensor(shapedata.std).to(device)
    with torch.no_grad():
        for i, sample_dict in enumerate(tqdm(dataloader_test)):
            tx = sample_dict['points'].to(device)
            prediction = model(tx)  
            if i==0:
                predictions = copy.deepcopy(prediction)
            else:
                predictions = torch.cat([predictions,prediction],0) 
                
            x_recon = prediction[:,:-1]
            x = tx[:,:-1]
            l1_loss+= torch.mean(torch.abs(x_recon-x))*x.shape[0]/float(len(dataloader_test.dataset))
            
            x_recon = (x_recon * shapedata_std + shapedata_mean) * mm_constant
            x = (x * shapedata_std + shapedata_mean) * mm_constant
            l2_loss+= torch.mean(torch.sqrt(torch.sum((x_recon - x)**2,dim = 2)))*x.shape[0]/float(len(dataloader_test.dataset))
            
        predictions = predictions.cpu().numpy()
        l1_loss = l1_loss.item()
        l2_loss = l2_loss.item()
    
    return predictions, l1_loss, l2_loss
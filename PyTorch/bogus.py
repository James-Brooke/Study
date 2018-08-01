import torch
import torch.nn.functional as F
from torch.autograd import Variable


class atk:
    """
    Adversarial attacks
    """
    def __init__(self):

        self.grads = {} 

    def save_grad(self, name):  #closure for use as a hook in fgsm attack - otherwise gradients can't be obtained for images. 
        def hook(grad):
            self.grads[name] = grad
        return hook

    def fgsm(self, model, x, y, eps=0.3, x_val_min=0, x_val_max=1): #https://arxiv.org/pdf/1412.6572.pdf

        criterion = torch.nn.CrossEntropyLoss()
        
        x_adv = Variable(x.data, requires_grad=True).cuda() #clean image
        x_adv.register_hook(self.save_grad('x_adv'))

        h_adv = model(x_adv) #clean pred
        
        cost = criterion(h_adv, y.cuda()) 

        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)

        cost.backward()

        x_adv = x_adv + (eps*self.grads['x_adv'].sign())
        x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
        
        
        return x_adv

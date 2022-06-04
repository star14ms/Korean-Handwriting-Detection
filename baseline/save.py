import os
import torch


model_dir = 'save/' 

def save_model(model_name, model, optimizer, scheduler):
    os.makedirs(os.path.join(model_dir),exist_ok=True)
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(state, os.path.join(model_dir, model_name + '.pth'))
    print(os.path.join(model_dir, model_name + '.pth'))
    print('model saved')    

    
def load_model(model_name, model, optimizer=None, scheduler=None):
    state = torch.load(os.path.join(model_dir, model_name + '.pth'))
    model.load_state_dict(state['model'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(state['scheduler'])
    print('model loaded')

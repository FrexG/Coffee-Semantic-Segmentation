from tqdm import tqdm
import os
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from datetime import datetime

class TrainModel:
    """ Model Training class
        returns -> None
        `Parameters`
        ----------
        `model`: torch model
        `train_loader`: torch DataLoader instance for the training data
        `test_loader`: torch DataLoader instance for the testing data
        `train_data_len`: length of the training data
        `test_data_len`: length of the testing data
        `device`: acceleration device (cpu,cuda or mps)
    """
    CURRENT_EPOCH = 0
    CURRENT_VAL_LOSS = float('inf')
    
    def __init__(self,model,test_loader:DataLoader,train_loader:DataLoader,train_data_len:int,test_data_len:int,device:torch.device=torch.device("cpu")) -> None:
        """ Initialization """
        
        self.model = model
        self.test_loader = test_loader
        self.train_loader = train_loader
        self.train_data_len = train_data_len
        self.test_data_len = test_data_len
        self.device = device

    def train(self,*,criterion,optimizer,batch_size,num_epoch:int=1):
        """ A Function to train a model
            returns -> trained model

            `Parameters`
            ----------
            `criterion`: The loss function
            `opt`: Gradient optimization function of type 'torch.optim.'
            `num_epoch`: The number of epoch for training

        """
        train_step = self.train_data_len // batch_size
        test_step = self.test_data_len // batch_size
        # Start training loop
        history = {'training_dice_loss':[],'test_dice_loss':[],'epochs':[]}
        # Model to device
        model = self.model.to(self.device)
        scheduler = ReduceLROnPlateau(optimizer,"min",patience=5)
        for epoch in range(num_epoch):
            # set the model to train mode

            model.train()
            # initialize total train and test loss
            total_train_loss = 0
            total_train_dice_loss = 0
            total_test_loss = 0
            total_test_dice_loss = 0

            progress_bar = tqdm(enumerate(self.train_loader),total=len(self.train_loader),desc='Training')
            for i,(images,masks) in progress_bar:
                # Move the images and masks to device
                images = images.to(self.device)
                masks = masks.to(self.device)
                # Perform forward pass and calculate the loss
                preds = model(images)
            
                loss = criterion(preds,masks)
                dice_loss = criterion(preds,masks)
                
                # zero grand and back-propagate
                optimizer.zero_grad()
                dice_loss.backward()
                optimizer.step()
                total_train_loss += loss
                total_train_dice_loss += dice_loss

                progress_bar.set_description(f"Epoch {epoch+1}/{num_epoch}")
                progress_bar.set_postfix(dice_loss=round(total_train_dice_loss.data.item()/(i+1),4),dice_coeff=round(1-(total_train_dice_loss.data.item()/(i+1)),4))
                progress_bar.update()
            
                # Evaluate the model on test set
            with torch.no_grad():
                #set the model to evaluation mode
                model.eval()
                eval_progress_bar = tqdm(enumerate(self.test_loader),total=len(self.test_loader),desc='Evaluation')
                for i,(images,masks) in eval_progress_bar:
                    # Move the images and masks to device
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                        # Perform forward pass and calculate the loss
                    preds = model(images)
                    
                    loss = criterion(preds,masks)
                    dice_loss = criterion(preds,masks)
                    total_test_loss += loss
                    total_test_dice_loss += dice_loss

                    eval_progress_bar.set_description(f"Evaluation")
                    eval_progress_bar.set_postfix(avg_dice_val_loss=round(total_test_dice_loss.data.item() / test_step,4),avg_val_dice_coeff=round(1-(total_test_dice_loss.data.item() / test_step),4))
                    eval_progress_bar.update()
            scheduler.step(total_test_dice_loss)
            # save history
            history['epochs'].append(epoch)
            history['training_dice_loss'].append(round(total_train_dice_loss.data.item() / train_step,4))
            history['test_dice_loss'].append(round(total_test_dice_loss.data.item() / test_step,4))

            if self.early_stopping(model,total_test_dice_loss,epoch):
                # Stop training
                print(f"[Info] Early Stopping")
                print(f"Stopped!!")
                
                # close tensorboard
                #self.writer.close()
        # Release GPU memory
        print(f"Free GPU ......")
        return model,history

    def early_stopping(self,model,val_loss,epoch,patience=10,min_delta=0.01):
        """ Helper method for model training early stopping """
        if val_loss < self.CURRENT_VAL_LOSS and self.CURRENT_VAL_LOSS - val_loss >= min_delta:
            # Val loss improved -> save model
            print(f"[Info] Val loss improved from {self.CURRENT_VAL_LOSS} to {val_loss}")
            self.save_model(model)
            self.CURRENT_VAL_LOSS = val_loss
            self.CURRENT_EPOCH = epoch
            return False
        if val_loss >= self.CURRENT_VAL_LOSS and epoch - self.CURRENT_EPOCH >= patience:
            ## Stop training
            return True

        return False

    def save_model(self,model):
        date_postfix = datetime.now().strftime("%Y-%m-%d")
        model_name = f'elunet_healthy_leaf_{date_postfix}.pth'
        save_path = "../weights/multi_class"
        
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        else:
            print(f"[INFO] Saving model to {os.path.join(save_path,model_name)}")
            torch.save(model.state_dict(),os.path.join(save_path,model_name))
           
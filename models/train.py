from tqdm import tqdm
import torch

def train(*,model,criterion,opt,train_loader,test_loader,train_data_len,test_data_len,device,batch_size,num_epoch:int=1):
    """ A Function to train a model
        returns -> trained model

        `Parameters`
        ----------
        `model`: torch model
        `criterion`: The loss function
        `opt`: Gradient optimization function of type 'torch.optim.'
        `train_loader`: torch DataLoader instance for the training data
        `test_loader`: torch DataLoader instance for the testing data
        `num_epoch`: The number of epoch for training

    """
    train_step = train_data_len // batch_size
    test_step = test_data_len // batch_size
    # Start training loop
    history = {'training_dice_loss':[],'test_dice_loss':[],'epochs':[]}
    # Model to device
    model = model.to(device)
    for epoch in range(num_epoch):
        # set the model to train mode

        model.train()
        # initialize total train and test loss
        total_train_loss = 0
        total_train_dice_loss = 0
        total_test_loss = 0
        total_test_dice_loss = 0

        progress_bar = tqdm(enumerate(train_loader),total=len(train_loader),desc='Training')
        for i,(images,masks) in progress_bar:
            # Move the images and masks to device
            images = images.to(device)
            masks = masks.to(device)
            # Perform forward pass and calculate the loss
            preds = model(images)
        
            loss = criterion(preds,masks)
            dice_loss = criterion(preds,masks)
            
            # zero grand and back-propagate
            opt.zero_grad()
            dice_loss.backward()
            opt.step()
            total_train_loss += loss
            total_train_dice_loss += dice_loss

            progress_bar.set_description(f"Epoch {epoch+1}/{num_epoch}")
            progress_bar.set_postfix(loss=round(total_train_loss.data.item()/(i+1),4),dice_loss=round(total_train_dice_loss.data.item()/(i+1),4),dice_coeff=round(1-(total_train_dice_loss.data.item()/(i+1)),4))
            progress_bar.update()
        
            # Evaluate the model on test set
        with torch.no_grad():
            #set the model to evaluation mode
            model.eval()
            eval_progress_bar = tqdm(enumerate(test_loader),total=len(test_loader),desc='Evaluation')
            for i,(images,masks) in eval_progress_bar:
                # Move the images and masks to device
                images = images.to(device)
                masks = masks.to(device)
                    # Perform forward pass and calculate the loss
                preds = model(images)
                
                loss = criterion(preds,masks)
                dice_loss = criterion(preds,masks)
                total_test_loss += loss
                total_test_dice_loss += dice_loss

                eval_progress_bar.set_description(f"Evaluation")
                eval_progress_bar.set_postfix(avg_loss=round(total_test_loss.data.item() / test_step,4),avg_dice_val_loss=round(total_test_dice_loss.data.item() / test_step,4),avg_val_dice_coeff=round(1-(total_test_dice_loss.data.item() / test_step),4))
                eval_progress_bar.update()
    # save history
    history['epochs'].append(epoch)
    history['training_dice_loss'].append(round(total_train_dice_loss.data.item() / train_step,4))
    history['test_dice_loss'].append(round(total_test_dice_loss.data.item() / test_step,4))

    """ # save model every 10 epochs
    if (epoch + 1) % 10 == 0:
        model_path = 'unet_torch_model.pth'
        print(f"[INFO] Saving model to {model_path}")
        torch.save(model.state_dict(),model_path) """

    return model,history
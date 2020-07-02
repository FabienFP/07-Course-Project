### 07-Course-Project

# PYTORCH CNN with Great Regularization and using GPU

This is a reflexion for a course project 
https://jovian.ml/forum/c/pytorch-zero-to-gans/18
Thanks to @aakashns for is course

# The modeling objective clearly

The DataSet is not very important. 
Here, we select Pikachu Web Scappring image from Google 

It s a Classification problem with (5)) classes : bulbasaur, charmander, mewtwo, pikachu, squirtle

We use an pretrain model like ResNet34 and change the last layer.

Improve with 
        - Data Augmentation
        - Drop Out in the last layer
        - Fit with Regularization to improve training and reduce overfitting : 
                1- Learning Rate Scheduling
                2- Weight decay
                3- Gradient clipping



# STEP 1 : Exploring the data
Data : 
The dataset is not very large and we can make easly an overfitting model if we do simples things
They are 5 classes.
The images are in png format with transparency and we have to transform in jpg
Image dim are not equal
Data are list in repotories of data (not shuffle)

### There is only one label per image : it's a single classification problem

# STEP 2 : Augment and transform
We augment and transform using Tranform.compose Pytorch function 
We use de imagenets_stats to normalize . 
#### issu 1, we could estimate the stats of our dataset 
#### issu 2, we could find a function to train - valid split the train data set 
(here we make a function to split randomly the dataset and create new file with copie of images)
This is note a simple way. But it's right. 
Train has an augmente data transform, and Val has a simple transform (like test dataset).
Usely, we use a numpy array that we can split with a simple function,
but we can't load a numpy array in ImageFolder()


# STEP 3: base model
We choice the Resnet34 model pre-train


# SETP 4 : improve the model
improve with 
        - Data Augmentation
        - Drop Out in the last layer
        - Fit with Regularization to improve training and reduce overfitting : 
                1- Learning Rate Scheduling
                2- Weight decay
                3- Gradient clipping

# STEP 5 : Train

We do 3 runs 

## freeze or not ?
We usely freeze and unfreeze fisrt layers but, here, the result is not good, so we unfreeze the model

## 1st run
epochs = 10 #10
max_lr = 0.005
grad_clip = None
weight_decay = 0
opt_func = torch.optim.Adam
For the 1st run, regularization is note efficient, so we don't use weight_decay and grad_clip

##### result
Epoch [9], last_lr: 0.00000, train_loss: 1.5274, val_loss: 1.4842, val_acc: 0.4102
CPU times: user 30.4 s, sys: 11.3 s, total: 41.7 s
Wall time: 14min 58s

## 2nd run
epochs = 20 #20
max_lr = 0.001
grad_clip = 0.05
weight_decay = 1e-4
opt_func = torch.optim.Adam

##### result
Epoch [19], last_lr: 0.00000, train_loss: 1.3342, val_loss: 1.2779, val_acc: 0.6078
CPU times: user 1min 1s, sys: 22.2 s, total: 1min 24s
Wall time: 29min 37s

## 3rd run
epochs = 20 #20
max_lr = 0.001
grad_clip = 0.05
weight_decay = 1e-4
opt_func = torch.optim.Adam

##### result
Epoch [19], last_lr: 0.00000, train_loss: 1.3220, val_loss: 1.2570, val_acc: 0.6101
CPU times: user 1min 2s, sys: 21.7 s, total: 1min 24s
Wall time: 29min 52s

# improve the result
Training loss is style > Validation loss , so we could run others epochs


# improve the code
3 issues are interesting to developpe (cf up)

#### issu 1, we could estimate the stats of our dataset 
#### issu 2, we could find a function to train - valid split the train data set 
#### issu 3 : find a function to choice best param (weight_decay and max_lr) like kfolds in keras function


# STEP 5 : Make some prediction
a predict_image() function is tested with 2 exemples 

# Save the model
The saved model function is tested by 
Evaluate on test dataset
Save the model weights
load the model
Evaluate with de model_load



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
                1- Learning Rate Schduling
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
#### issu 2, we could find a function to train - valid split the train data set (here we do the separation manually)


# STEP 3: base model
We choice the Resnet50 model pre-train


# SETP 4 : improve the model
mprove with 
        - Data Augmentation
        - Drop Out in the last layer
        - Fit with Regularization to improve training and reduce overfitting : 
                1- Learning Rate Schduling
                2- Weight decay
                3- Gradient clipping

# STEP 5 : Train

We do 2 run with freeze and unfreeze weights of the pre-train model with :

epochs = 15
max_lr = 0.01
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam

We note have the time to change the hyperparameter
#### issu 3 : find a function to choice best param like kfolds in keras function
3 functions plot the progress of the training : plot_scores(), plot_losses(), plot_lrs()

# STEP 5 : Make some prediction
a predict_image() function is tested with 2 exemples 

# Save the model
The saved model function is tested by 
Evaluate on test dataset
Save the model weights
load the model
Evaluate with de model_load

# Conclusion
3 issues are interesting to developpe (cf up)
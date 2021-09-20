# Training a NN to 99% accuracy on MNIST in 0.76 seconds

A quick study on how fast you can reach 99% accuracy on MNIST with a single laptop. Our answer is 0.76 seconds, reaching 99% accuracy in just one epoch of training. This is more than 200 times faster than the default training code from Pytorch. To see the final results, check 8_Final_00s76.ipynb. If you're interested in the process read on below for a step by step description of changes made. 

The repo is organized into jupyter notebooks, showing a chronological order of changes required to go from 
initial Pytorch tutorial that trains for 3 minutes to less than a second of training time
on a laptop with GeForce GTX 1660 Ti GPU. I aimed for a coordinate ascent like procedure, changing only one thing at a time to make sure we understand what is the source of improvements each time, but sometimes I bunched up correlated or small changes.

### Requirements

Python3.x and Pytorch 1.8 (most likely works with >= 1.3). For fast times you'll need Cuda and a compatible GPU.


## 0_Pytorch_initial_2m_52s.ipynb: Starting benchmark

First we need to benchmark starting performance. This can be found in the file 0_Pytorch_initial_2m_52s.ipynb.
Note the code downloads the dataset if not already present so reporting second run time. Trains for 14 epochs each run, average accuracy of two runs is 99.185% on test set, and the mean runtime is **2min 52s ± 38.1ms**.

## 1_Early_stopping_57s40.ipynb: Stop early

Since our goal is to reach only 99% accuracy, we don't need the full training time. Our first modification is to simply stop training after the epoch we hit 99% test accuracy. This is typically reached within 3-5 epochs with average final accuracy of 99.07%, cutting training time to around a third of the original at **57.4s ± 6.85s**. 

## 2_Smaller_NN_30s30.ipynb: Reduce network size

Next we employ the trick of reducing both network size and regularization to speed up convergence. This is done by adding a 2x2 max pool layer after the first conv layer, reducing parameters in our fully connected layers by more than 4x. To compensate we also remove one of the 2 dropout layers. This reduces number of epochs we need to converge to 2-3, and training time to **30.3s ± 5.28s**.

## 3_Data_loading_07s31.ipynb: Optimize Data Loading!

This is probably the biggest and most surprising time save of this project. Just by better optimizing the data loading process we can save 75% of the entire training run time. It turns out that torch.utils.data.DataLoader is really inefficient for small datasets like MNIST, instead of reading it from the disk one batch at a time we can simply load the entire dataset into GPU memory at once and keep it there. To do this we save the entire dataset with the same processing we had before onto disk in a single pytorch array using data_loader.save_data(). This takes around 10s and is not counted in the training time as it has to be done only once. With this optimization, our average training time goes down to **7.31s ± 1.36s**.

## 4_128_Batch_04s66.ipynb: Increase batch size

Now that we have optimized data loading, increasing batch size can significantly increase the speed of training. Simply increasing the batch size from 64 to 128 reduces our average train time to **4.66s ± 583ms**.

## 5_Onecycle_lr_03s14.ipynb: Better learning rate schedule

For this step, we turn our looks to to the learning rate schedule. Previously we used an exponential decay where after each epoch lr is multiplied by 0.7. We replace this by [Superconvergence](https://arxiv.org/pdf/1708.07120.pdf) also known as OneCycleLR, where the learning starts close to 0 and is linearly(or with cosine schedule) increased to to its peak value at the middle of training and slowly lowered down to zero again in the end. This allows using much higher learning rates than otherwise. We used peak LR of 4.0, 4 times higher than the starting lr used previously. The network reaches 99% in 2 epochs every time now, and this takes our training time down to **3.14s ± 4.72ms**.  

## 6_256_Batch_02s31.ipynb: Increase batch size, again

With our better lr schedule we can once more double our batch size without hurting performance much. Note this time around it doesn't reach 99% on all random seeds but I count it as a success as long I'm confident the mean accuracy is greater than 99%. This is because Superconvergence requires a fixed length training and we can't quarantee every seed works. This cuts our training time down to **2.31s ± 23.2ms**.

## 7_Smaller_NN2_01s74.ipynb: Remove dropout and reduce size, again

Next we repeat our procedure from step 2 once again, remove the remaning dropout layer and compensate by reducing the width of our convolutional layers, first to 24 from 32 and second to 32 from 64. This reduces the time to train an epoch, and even nets us with increased accuracy, averaging around 99.1% after two epochs of training. This gives us mean time of **1.74s ± 18.3ms**.

## 8_Final_00s76.ipynb: Tune everything

Now that we have a fast working model and we have grabbed most of the low hanging improvements, it is time to dive into final finetuning. To start off, we simply move our max pool operations before the ReLU activation, which doesn't change the network but saves us a bit of compute. 

The next changes were the result of a large search operation, where I tried a number of different things, optimizing one hyperparameter at a time. For each change I trained on 30 different seeds and measured what gives us the highest mean accuracy. 30 seeds was necessary to make statistically significant conclusions on small changes, and it is worth noting training 30 seeds took less than a minute at this point. Higher accuracy can then be translated into faster times by cutting down on the number of epochs.

First I actually made the network bigger in select places that didn't slow down performance too much. The kernel size of first convolutional layer was incresed from 3 to 5, and the final fully connected layer increased from 128 to 256. 

Next, it was time to change the optimizer. I found that with proper hyperparameters, Adam actually outperforms Adadelta which we had used so far. The hyperparameters I changed from default are learning rate of 0.01(default 0.001), beta1 of 0.7(default 0.9) and bata2 of 0.9(default 0.999). 

All of this lead to a large boost in accuracy(99.245% accuracy after 2 epochs), which I was able to finally trade into faster training times by cutting training down to just one epoch! Our final result is 99.04% mean accuracy in just **762ms ± 24.9ms**.

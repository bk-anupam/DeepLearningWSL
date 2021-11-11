import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from sklearn import model_selection
from PIL import Image
import torchvision.transforms.functional as F

def get_imgs_mean_stddev(imgs, axis=None):    
    """Get the mean and standard deviation for images in a dataset / mini-batch. It works by loading 
    entire dataset/mini batch in memory and using pytorch built in mean and std functions
    img batch is of shape BS * C * H * W 
    where BS = batch_size or no of training samples 
    C = 3 ( RGB channels ), H = height of image matrix, W = width of image matrix
    Args:
        imgs ([2d or 3d numpy array]): images in collection (with no to_tensor transformation applied)
        axis ([tuple of ints], optional): Axis along which mean and std dev is to be calculated.
        Defaults to None.
    Returns:
        [tuple]: tuple of tensors with mean and std.dev. of the imgs
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    to_tensor = ToTensor()
    img_tensor_arr = [to_tensor(img) for img in imgs]    
    # stack will arrange the tensors one over the other with dim=0 being the new dimension that  
    # stores the number of tensors stacked. This new dimension can be placed at any index
    img_tensor_arr = torch.stack(img_tensor_arr)
    img_tensor_arr = img_tensor_arr.to(device)
    if axis is None:
        axis = (0, 2, 3)
    mean, std = torch.mean(img_tensor_arr, axis=axis), torch.std(img_tensor_arr, axis=axis)
    return mean.cpu(), std.cpu()

import tqdm

# https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/03/08/image-mean-std.html
# https://www.thoughtco.com/sum-of-squares-formula-shortcut-3126266
def get_imgs_mean_stddev_streaming(dl_img, axis=None):    
    """Get the mean and standard deviation for images in a dataset / mini-batch. It should be used when
    dataset can't fit in memory, so instead a different is used 
    img batch is of shape BS * C * H * W 
    where BS = batch_size or no of training samples 
    C = 3 ( RGB channels ), H = height of image matrix, W = width of image matrix
    Args:
        dl_imgs ([DataLoader]): image data loader
        axis ([tuple of ints], optional): Axis along which mean and std dev is to be calculated.
        Defaults to None.
    Returns:
        [tuple]: tuple of tensors with mean and std.dev. of the imgs
    """
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    # sum of pixel values along RGB channels
    psum = torch.Tensor([0.0, 0.0, 0.0])
    # sum of squares of pixel values along RGB channels
    psum_sq = torch.Tensor([0.0, 0.0, 0.0])        
    num_img = 0    
    img_h, img_w = 0, 0    
    count = 0
    for img, label in tqdm.tqdm(dl_img): 
        if count == 0:            
            img_h = img.shape[2]       
            img_w = img.shape[3]
        num_img += img.shape[0]            
        psum += img.sum(axis=[0, 2, 3])        
        img_sq = img.square()
        psum_sq += img_sq.sum(axis=[0, 2, 3])
        count += 1
    # pixel count of single img (index 1 is the height and index 2 is width of img)
    img_pixel_count = img_h * img_w      
    total_pixel_count = num_img * img_pixel_count   
    # mean of pixel values across the dataset        
    total_mean = psum / total_pixel_count    
    # variance of pixel values across the dataset
    total_var = (psum_sq / total_pixel_count) - (total_mean.square())    
    total_std = torch.sqrt(total_var)
    return total_mean, total_std


# for a training and label data in form of numpy arrays, return a fold_index array whose elements
# represent the fold index. The length of this fold_index array is same as length of input dataset
# and the unique fold values represent the items to be used for validation in the corresponding
# cross validation iteration with rest of the items being used for training (typical ration being 80:20)
def get_skf_index(num_folds, X, y):
    skf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state = 42)
    train_fold_index = np.zeros(len(y))
    for fold, (train_index, val_index) in enumerate(skf.split(X=X, y=y)):
        train_fold_index[val_index] = [fold + 1] * len(val_index)
    return train_fold_index        

# split the training dataframe into kfolds for cross validation. We do this before any processing is done
# on the data. We use stratified kfold if the target distribution is unbalanced
def strat_kfold_dataframe(df, target_col_name, num_folds=5):
    # we create a new column called kfold and fill it with -1
    df["kfold"] = -1
    # randomize of shuffle the rows of dataframe before splitting is done
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    # get the target data
    y = df["target"].values
    skf = model_selection.StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    for fold, (train_index, val_index) in enumerate(skf.split(X=df, y=y)):
        df.loc[val_index, "kfold"] = fold    
    return df        

# display images along with their labels from a batch where images are in form of numpy arrays 
# if predictions are provided along with labels, these are displayed too
def show_batch(img_arr, label_arr, img_index, num_rows, num_cols, predict_arr=None):
    fig = plt.figure(figsize=(9, 6))
    for index, img_index in enumerate(img_index):  # list first 9 images
        img, lb = img_arr[img_index], label_arr[img_index]
        ax = fig.add_subplot(num_rows, num_cols, index + 1, xticks=[], yticks=[])
        if isinstance(img, torch.Tensor):
            img = img.detach().numpy()
        if isinstance(img, np.array):
            # the image data has RGB channels at dim 0, the shape of 3, 64, 64 needs to be 64, 64, 3 for display
            img = img.transpose(1, 2, 0)
            ax.imshow(Image.fromarray(np.uint8(img)).convert('RGB'))        
        title = f"Actual: {lb}"
        if predict_arr: 
            title += f", Pred: {predict_arr[img_index]}"        
        ax.set_title(title)    


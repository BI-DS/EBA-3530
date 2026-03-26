import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import glob
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tensorflow.keras.utils import image_dataset_from_directory as load_images
from time import time
import pickle
import subprocess

address_tr = 'https://www.dropbox.com/scl/fi/w7hjg8ucehnjfv1re5wzm/mnist_color.pkl?rlkey=ya9cpgr2chxt017c4lg52yqs9&st=ev984mfc&dl=0'
address_te = 'https://www.dropbox.com/scl/fi/w08xctj7iou6lqvdkdtzh/mnist_color_te.pkl?rlkey=xntuty30shu76kazwhb440abj&st=u0hd2nym&dl=0'
address_y_te = 'https://www.dropbox.com/scl/fi/fkf20sjci5ojhuftc0ro0/mnist_color_y_te.npy?rlkey=fshs83hd5pvo81ag3z209tf6v&st=99z1o18q&dl=0'

def get_url_data(extention = 'pkl',file_name='mmnist'):
    try:
        subprocess.run(["wget", "-q", address_tr, "-O", '../data/'+file_name+'_tr.'+extention], check=True)
        subprocess.run(["wget", "-q", address_te, "-O", '../data/'+file_name+'_te.'+extention], check=True)
        print("Files downloaded successfully")
    except Exception as e:
        print(f"Error downloading file: {e}")

class MMNISTDataset(tf.keras.utils.Sequence):
    def __init__(self, batch_size = 500,
                       shuffle=False,
                       normalize=False,
                       folder_path = '../data/MMNIST',
                       train = True,
                       num_modalities=5
                ):
        super().__init__()
        self.num_modalities = num_modalities
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.normalize = normalize
        self.train = train

        if os.path.isfile('../data/mmnist_tr.pkl') and os.path.isfile('../data/mmnist_te.pkl'):
            print('Data files already exist!')
        else:
            print('Downloading data...')
            get_url_data()
            
        if train:
            self.folder_path = os.path.join(folder_path,'train')
        else:
            self.folder_path = os.path.join(folder_path,'test')

        self.unimodal_datapaths=[]
        for i in range(self.num_modalities):
            self.unimodal_datapaths.append(os.path.join(self.folder_path,'m'+str(i)))

    def get_data(self):
        all_ds=[]
        for f in self.unimodal_datapaths:
            ds = load_images(f, labels=None, batch_size=self.batch_size, shuffle=self.shuffle, image_size=(28,28))
            if self.normalize:
                normalization_layer = tf.keras.layers.Rescaling(1./255)
                ds = ds.map(lambda x: normalization_layer(x), num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.cache()
            ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            all_ds.append(ds)

        return all_ds

    def get_labels(self):
        files = glob.glob(os.path.join(self.unimodal_datapaths[0], "*.png"))
        labels = [int(file.split(".")[-2]) for file in files]
        
        return tf.keras.utils.to_categorical(np.array(labels), num_classes=10)

    def _get_modalities(self):
        #returns a list of dataloader
        modalities = self.get_data()
        # create dict to save the list of data loaders
        modalities_dict = dict()
        
        for i in range(len(modalities)):
            name = 'm'+str(i)
            modalities_dict[name]=modalities[i]
       
        self.dataloader_zip = tf.data.Dataset.zip(modalities_dict)
        self.modalities = modalities
    
    def get_modalities(self, as_dict=False):   
        self._get_modalities()
        if as_dict:
            all_batches = []
            
            print('going throug imgs...')
            for i, batch in enumerate(self.dataloader_zip):
                all_batches.append(np.array(list(batch.values())))
                
                if (i+1)%15==0:
                    print('{} batches already done!'.format(i+1))

            data = np.concatenate(all_batches,1) # (num_modalites, no_obs, 28, 28, 3)
            print(data.shape)

            data_dict = dict()
            for i in range(data.shape[0]):
                data_dict['m'+str(i)] = data[i]
            return data_dict
        else:
            return self.dataloader_zip
    
    def load_data(self, data_dir='../data', dset='mmnist'):
        if self.train:
            file_name = dset+'_tr.pkl'
        else:
            file_name = dset+'_te.pkl'
        
        with open(os.path.join(data_dir,file_name), 'rb') as f:
            data_dir = pickle.load(f) # this file contains all modalities
        
        new_dir = dict()
        for i in range(self.num_modalities):
            mod_name = 'm'+str(i)
            new_dir[mod_name] = data_dir[mod_name]
        
        #target = self.get_labels() 
        return new_dir


if __name__ == "__main__":
    ds = MMNISTDataset(train=True, normalize=True, batch_size=256, shuffle=False, num_modalities=5, folder_path='../data/PolyMNIST')

    start = time()
    data_dir = ds.get_modalities(as_dict=True)
    with open('../data/polymnist_te.pkl','wb') as f:
        pickle.dump(data_dir, f,  protocol=pickle.HIGHEST_PROTOCOL)
    print('elapseed time',time()-start)

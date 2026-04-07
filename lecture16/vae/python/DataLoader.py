import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle
import tensorflow as tf
import numpy as np
import subprocess

class DataLoader:
    def __init__(self, dset):
        self.dset = dset
        if dset == 'mnist_bw':
            self.address_tr = 'https://www.dropbox.com/scl/fi/fjye8km5530t9981ulrll/mnist_bw.npy?rlkey=ou7nt8t88wx1z38nodjjx6lch&st=5swdpnbr&dl=0'
            self.address_te = 'https://www.dropbox.com/scl/fi/dj8vbkfpf5ey523z6ro43/mnist_bw_te.npy?rlkey=5msedqw3dhv0s8za976qlaoir&st=nmu00cvk&dl=0'
            self.address_y_te = 'https://www.dropbox.com/scl/fi/8kmcsy9otcxg8dbi5cqd4/mnist_bw_y_te.npy?rlkey=atou1x07fnna5sgu6vrrgt9j1&st=m05mfkwb&dl=0'
            self.file_name = 'mnist_bw'
        elif dset == 'mnist_color':
            self.address_tr = 'https://www.dropbox.com/scl/fi/w7hjg8ucehnjfv1re5wzm/mnist_color.pkl?rlkey=ya9cpgr2chxt017c4lg52yqs9&st=ev984mfc&dl=0'
            self.address_te = 'https://www.dropbox.com/scl/fi/w08xctj7iou6lqvdkdtzh/mnist_color_te.pkl?rlkey=xntuty30shu76kazwhb440abj&st=u0hd2nym&dl=0'
            self.address_y_te = 'https://www.dropbox.com/scl/fi/fkf20sjci5ojhuftc0ro0/mnist_color_y_te.npy?rlkey=fshs83hd5pvo81ag3z209tf6v&st=99z1o18q&dl=0'
            self.file_name = 'mnist_color'
    
    def get_url_data(self):
        if self.file_name == 'mnist_color':
            extention = 'pkl'
        else:
            extention = 'npy'
        try:
            subprocess.run(["wget", "-q", self.address_tr, "-O", '../data/'+self.file_name+'_tr.'+extention], check=True)
            subprocess.run(["wget", "-q", self.address_te, "-O", '../data/'+self.file_name+'_te.'+extention], check=True)
            subprocess.run(["wget", "-q", self.address_y_te, "-O", '../data/'+self.file_name+'_y_te.npy'], check=True)
            print("Files downloaded successfully")
        except Exception as e:
            print(f"Error downloading file: {e}")
    
    def load_data(self, modality='m1'):
        if os.path.isfile('../data/'+self.file_name+'_y_te.npy'):
            print('Data files already exist!')
        else:
            print('Downloading data...')
            self.get_url_data()
        y_te = np.load('../data/'+self.file_name+'_y_te.npy')
        if self.dset=='mnist_bw':
            data_tr = np.load('../data/'+self.file_name+'_tr.npy')
            data_te = np.load('../data/'+self.file_name+'_te.npy')
        elif self.dset=='mnist_color':
            with open('../data/'+self.file_name+'_tr.pkl', 'rb') as f:
                data_tr = pickle.load(f)
            with open('../data/'+self.file_name+'_te.pkl', 'rb') as f:
                data_te = pickle.load(f)
            data_tr = data_tr[modality]
            data_te = data_te[modality]
        self.data_tr = data_tr
        self.data_te = data_te
        self.y_te = y_te

    def preprocessing(self, data):
        data = data.reshape(data.shape[0],data.shape[1]*data.shape[2])
        data = data/255.
        data = tf.cast(data,tf.float32)
        return data

    def tf_loader(self, batch_size=256, train=True):
        if train:
            if self.dset=='mnist_bw':
                data  = self.preprocessing(self.data_tr)
            else:
                data = self.data_tr
            BUFFER = data.shape[0]
            print('input data shape',data.shape)
            tr_data = tf.data.Dataset.from_tensor_slices(data).shuffle(BUFFER).batch(batch_size)
            return tr_data
        else:
            if self.dset=='mnist_bw':
                data_te = self.preprocessing(self.data_te)
            else:
                data_te = self.data_te
            return data_te, self.y_te

if __name__=='__main__':
    d = DataLoader('mnist_color')
    d.get_url_data()

# # encoding: utf-8
# """
# @author:  sherlock
# @contact: sherlockliao01@gmail.com
# """

#train with naicval
import glob
import os.path as osp
import re
import warnings
import random
from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY

def split(full_list, shuffle=False, ratio=0.06):
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1,sublist_2

@DATASET_REGISTRY.register()
class Pengcheng(ImageDataset):
    """Market1501.

    Reference:
        Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: `<http://www.liangzheng.org/Project/project_reid.html>`_

    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """
    _junk_pids = [0, -1]
    dataset_dir = ''
    dataset_url = 'http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip'
    dataset_name = "pengcheng"

    def __init__(self, root='datasets', pengcheng=False, **kwargs):

        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'NAIC')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            print("")
        self.label_txt = osp.join(data_dir,'train/new_label.txt')
        self.train_dir = osp.join(data_dir, 'train/images')
        self.query_dir = osp.join(data_dir, 'market/query')
        self.gallery_dir = osp.join(data_dir, 'market/gallery')

        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]

        self.label = {}
        self.process_label(self.label_txt)
        self.label_new = {}
        self.process_newlabel(self.label)
        self.lab = []
        for i in self.label_new.keys():
            self.lab.append(i)

        quelis = []
        trainlis = []
        f=open('query.txt')
        for line in f:
            quelis.append(str(int(line)))
        f=open('train.txt')
        for line in f:
            trainlis.append(str(int(line)))

        train = self.process_newtrain(self.train_dir,trainlis)
        query = self.process_newquery(self.train_dir,quelis,que = True)
        gallery = self.process_newquery(self.train_dir,quelis,que = False)
        self.checkid(query,gallery)

        super(Pengcheng, self).__init__(train, query, gallery, **kwargs)


    def process_query(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
            data.append((img_path, pid, camid))

        return data

    def checkid(self,query,gallery):
        q = []
        g = []
        for i in query:
            q.append(i[1])
        for i in gallery:
            g.append(i[1])
        for i in q:
            if i not in g:
                print("not appear",i)
    def process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        data = []
        camid = 1
        for img_path in img_paths:
            img = img_path.split("/")[-1]
            if img in self.label:
                pid = self.label[img]
                if is_train:
                    pid = self.dataset_name + "_" + str(pid)
                data.append((img_path, pid,camid))

        return data

    def process_label(self,label_file):
        f = open(label_file,encoding="utf-8")

        while True:
            content = f.readline()
            if content == '':
                break
            img,id = content.strip().split(':')
            self.label[img]=id
        f.close()
    def process_newlabel(self,label_file):
        for key, val in label_file.items():
            a = []
            a.append(key)

            if val in self.label_new.keys():
                self.label_new[val].append(key)
            else:
                self.label_new[val] = a

    def process_newtrain(self,datadir,labellist):
        data = []
        for i in labellist:
            for k in self.label_new[i]:
                pid = self.dataset_name + "_" + str(i)
                data.append((osp.join(datadir,k),pid,1))
        return data
    def process_newquery(self,datadir,quelist,que=True):
        data = []
        
        if que:
            for i in quelist:               
                data.append((osp.join(datadir,self.label_new[i][0]),int(i),1))
        else:
            for i in quelist:
                for k in self.label_new[i][1:]:                   
                    data.append((osp.join(datadir,k),int(i),2))
        return data



if __name__ == "__main__":
    dataset=Pengcheng('/Users/lijiaqi/Downloads/fast-reid-master/datasets')
    print(dataset.label)


# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import os.path as osp
import re
import warnings

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY

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

        # self.root = osp.abspath(osp.expanduser(root))
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'pengcheng')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "bounding_box_train" under '
                          '"Market-1501-v15.09.15".')
        self.label_txt=osp.join(data_dir,'train/label.txt')
        self.train_dir = osp.join(data_dir, 'train/images')
        self.query_dir = osp.join(data_dir, 'market/query')
        self.gallery_dir = osp.join(data_dir, 'market/gallery')
        #self.extra_gallery_dir = osp.join(self.data_dir, 'images')
        #self.market1501_500k = market1501_500k

        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        # if self.market1501_500k:
        #     required_files.append(self.extra_gallery_dir)
        #self.check_before_run(required_files)
        self.count=0
        self.label = {}
        self.process_label(self.label_txt)

        train = self.process_dir(self.train_dir)
        #print("11",len(train))
        query = self.process_query(self.query_dir, is_train=False)
        gallery = self.process_query(self.gallery_dir, is_train=False)
        #if self.market1501_500k:
        #    gallery += self.process_dir(self.extra_gallery_dir, is_train=False)
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



    def process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        #pattern = re.compile(r'([-\d]+)_c(\d)')
        data = []
        camid=1
        for img_path in img_paths:
            img=img_path.split("/")[-1]
            if img in self.label:
                pid=self.label[img]
            # pid, camid = map(int, pattern.search(img_path).groups())
            # if pid == -1:
            #     continue  # junk images are just ignored
            # assert 0 <= pid <= 1501  # pid == 0 means background
            # assert 1 <= camid <= 6
            # camid -= 1  # index starts from 0
                if is_train:
                    pid = self.dataset_name + "_" + str(pid)
                data.append((img_path, pid,camid))

        return data

    def process_label(self,label_file):
        f=open(label_file,encoding="utf-8")

        while True:
            #self.count=self.count+1
            content=f.readline()
            if content=='':
                break
            img,id=content.strip().split(':')
            self.label[img]=id
        #print("count",count)
        f.close()

if __name__ == "__main__":
    dataset=Pengcheng('/Users/lijiaqi/Downloads/fast-reid-master/datasets')
    print(dataset.label)

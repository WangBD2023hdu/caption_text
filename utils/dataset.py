import torch
from torch.utils.data import Dataset
import json
import os


class BaseSet(Dataset):
    def __init__(self, type="train", max_length=100, text_path=None, use_np=False, img_path=None, knowledge=0):
        """
        Args:
            type: "train","val","test"
            max_length: the max_lenth for bert embedding
            text_path: path to annotation file
            img_path: path to img embedding. Resnet152(,2048), Vit B_32(,768), Vit L_32(, 1024)
            use_np: True or False, whether use noun phrase as relation matching node. It is useless in this paper.
            img_path:
            knowledge: 1 caption, 2 ANP, 3 attribute, 0 not use knowledge
        """
        self.type = type  # train, val, test
        self.max_length = max_length
        self.text_path = text_path
        self.img_path = img_path
        self.use_np = use_np
        with open(self.text_path) as f:
            self.dataset = json.load(f)
        if self.type == 'train':
            with open("/home/wbd/source_code/base_roberta_twitter/twitter/dataset_text/traindep.json") as f:
                self.dataset_train = json.load(f)
        if self.type == 'val':
            with open("/home/wbd/source_code/base_roberta_twitter/twitter/dataset_text/valdep.json") as f:
                self.dataset_val = json.load(f)
        if self.type == 'test':
            with open("/home/wbd/source_code/base_roberta_twitter/twitter/dataset_text/testdep.json") as f:
                self.dataset_test = json.load(f)
        # if self.type != "train":
        self.img_set = torch.load(self.img_path)
        self.knowledge = int(knowledge)

    def __getitem__(self, index):
        """

        Args:
            index:

        Returns:
            img: (49, 768). Tensor.
            text_emb: (token_len, 758). Tensor
            text_seq: (word_len). List.
            dep: List.
            word_len: Int.
            token_len: Int
            label: Int
            chunk_index: li

        """
        sample = self.dataset[index]

        # for val and test dataset, the sample[2] is hashtag label
        if self.type=='train':
            #text = sample[4]
            caption = sample[3].split(' ')
        else:
            #text = sample[5]
            caption = sample[4].split(' ')

        if self.type == 'train':
            label = self.dataset_train[index][2]
            text = self.dataset_train[index][3]
        elif self.type == 'val':
            label = self.dataset_val[index][3]
            text = self.dataset_val[index][4]
        else:
            label = self.dataset_test[index][3]
            text = self.dataset_test[index][4]
        twitter = text["token_cap"]
        dep = text["token_dep"]
        
        img = self.img_set[index]

        return img, twitter, dep, caption, label



    def __len__(self):
        """
            Returns length of the dataset
        """
        return len(self.dataset)


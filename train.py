# -*- coding: UTF-8 -*-
import argparse
from tqdm import tqdm
import os
from utils.logging.tf_logger import Logger
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import torch
from torch.nn import CrossEntropyLoss
import numpy as np

from model import KEHModel_without_know
from utils.data_utils import construct_edge_image
from utils.dataset import BaseSet
from utils.compute_scores import get_metrics, get_four_metrics
from utils.data_utils import PadCollate_without_know
import json
import re
from utils.data_utils import seed_everything

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-m', '--mode', type=str, default='train',
                    help="mode, {'" + "train" + "', '" + "eval" + "'}")
parser.add_argument('-p', '--path', type=str, default='saved_model path',
                    help="path, relative path to save model}")
parser.add_argument('-s', '--save', type=str, default='saved model',
                    help="path, path to saved model}")
parser.add_argument('-o', '--para', type=str, default='parameter.json',
                    help="path, path to json file keeping parameter}")
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--max_length', type=int, default=120)
parser.add_argument('--use_np', type=bool, default=False)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--weight_decay', type=float, default=0.005)
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--txt_input_dim', type=int, default=768)
parser.add_argument('--txt_out_size', type=int, default=200)
parser.add_argument('--img_input_dim', type=int, default=768)
parser.add_argument('--img_inter_dim', type=int, default=500)
parser.add_argument('--img_out_dim', type=int, default=200)
parser.add_argument('--cro_layers', type=int, default=6)
parser.add_argument('--cro_heads', type=int, default=5)
parser.add_argument('--cro_drop', type=float, default=0.5)
parser.add_argument('--txt_gat_layer', type=int, default=2)
parser.add_argument('--txt_gat_drop', type=float, default=0.5)
parser.add_argument('--txt_gat_head', type=int, default=2)
parser.add_argument('--txt_self_loops', type=bool, default=True)
parser.add_argument('--img_gat_layer', type=int, default=2)
parser.add_argument('--img_gat_drop', type=float, default=0.5)
parser.add_argument('--img_gat_head', type=int, default=2)
parser.add_argument('--img_self_loops', type=bool, default=True)
parser.add_argument('--img_edge_dim', type=int, default=0)
parser.add_argument('--img_patch', type=int, default=49)
parser.add_argument('--lambda', type=int, default=1)
parser.add_argument('--type_bmco', type=int, default=1)
parser.add_argument('--knowledge_type', type=int, default=0)
parser.add_argument('--know_max_length', type=int, default=20)
parser.add_argument('--know_gat_layer', type=int, default=2)
parser.add_argument('--know_gat_head', type=int, default=1)
parser.add_argument('--know_cro_layer', type=int, default=3)
parser.add_argument('--know_cro_head', type=int, default=5)
parser.add_argument('--know_cro_type', type=int, default=0)
parser.add_argument('--visualization', type=bool, default=False)
args = parser.parse_args()
with open(args.para) as f:
    parameter = json.load(f)
# parameter["mode"] = args.mode
# parameter["patience"] = args.patience
# parameter["batch_size"] = args.batch_size
# parameter["epochs"] = args.epochs
# parameter["max_length"] = args.max_length
# parameter["use_np"] = args.use_np
# parameter["seed"] = args.seed
# parameter["weight_decay"] = args.weight_decay
# parameter["lr"] = args.lr
# parameter["txt_input_dim"] = args.txt_input_dim
# parameter["txt_out_size"] = args.txt_out_size
# parameter["img_input_dim"] = args.img_input_dim
# parameter["img_inter_dim"] = args.img_inter_dim
# parameter["img_out_dim"] = args.img_out_dim
# parameter["cro_layers"] = args.cro_layers
# parameter["cro_heads"] = args.cro_heads
# parameter["cro_drop"] = args.cro_drop
# parameter["txt_gat_layer"] = args.txt_gat_layer
# parameter["txt_gat_drop"] = args.txt_gat_drop
# parameter["txt_gat_head"] = args.txt_gat_head
# parameter["txt_self_loops"] = args.txt_self_loops
# parameter["img_gat_layer"] = args.img_gat_layer
# parameter["img_gat_drop"] = args.img_gat_drop
# parameter["img_gat_head"] = args.img_gat_head
# parameter["img_self_loops"] = args.img_self_loops
# parameter["img_edge_dim"] = args.img_edge_dim
# parameter["img_patch"] = args.img_patch
# parameter["type_bmco"] = args.type_bmco
# parameter["knowledge_type"] = args.knowledge_type
# parameter["know_max_length"] = args.know_max_length
# parameter["know_gat_layer"] = args.know_gat_layer
# parameter["know_gat_head"] = args.know_gat_head
# parameter["know_cro_layer"] = args.know_cro_layer
# parameter["know_cro_head"] = args.know_cro_head
# parameter["know_cro_type"] = args.know_cro_type
# parameter["visualization"] = args.visualization
annotation_files = parameter["annotation_files"]
img_files = parameter["DATA_DIR"]
use_np = parameter["use_np"]
knowledge_type = 0
model = KEHModel_without_know(txt_input_dim=parameter["txt_input_dim"], txt_out_size=parameter["txt_out_size"],
                              img_input_dim=parameter["img_input_dim"],
                              img_inter_dim=parameter["img_inter_dim"],
                              img_out_dim=parameter["img_out_dim"], cro_layers=parameter["cro_layers"],
                              cro_heads=parameter["cro_heads"], cro_drop=parameter["cro_drop"],
                              txt_gat_layer=parameter["txt_gat_layer"], txt_gat_drop=parameter["txt_gat_drop"],
                              txt_gat_head=parameter["txt_gat_head"],
                              txt_self_loops=parameter["txt_self_loops"], img_gat_layer=parameter["img_gat_layer"],
                              img_gat_drop=parameter["img_gat_drop"],
                              img_gat_head=parameter["img_gat_head"], img_self_loops=parameter["img_self_loops"],
                              img_edge_dim=parameter["img_edge_dim"],
                              img_patch=parameter["img_patch"], lam=parameter["lambda"],
                              type_bmco=parameter["type_bmco"], visualization=parameter["visualization"])
print("Image Encoder", sum(p.numel() for p in model.img_encoder.parameters() if p.requires_grad))
print("Text Encoder", sum(p.numel() for p in model.txt_encoder.parameters() if p.requires_grad))
print("Interaction", sum(p.numel() for p in model.interaction.parameters() if p.requires_grad))
print("Alignment", sum(p.numel() for p in model.alignment.parameters() if p.requires_grad))
print("Total Params", sum(p.numel() for p in model.parameters() if p.requires_grad))

model.to(device=device)
# 0.05
# TODO optimizer
optimizer = optim.Adam(params=model.parameters(), lr=parameter["lr"], betas=(0.9, 0.999), eps=1e-8,
                       weight_decay=parameter["weight_decay"],
                       amsgrad=True)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=parameter["patience"], verbose=True)
# optimizer = optim.Adam(params=model.parameters(), lr=parameter["lr"], betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=True)

cross_entropy_loss = CrossEntropyLoss()
# cross_entropy_loss = CrossEntropyLoss(weight=torch.tensor([1,1.1]).cuda())
# args.path must be relative path
logger = Logger(model_name=parameter["model_name"], data_name='twitter',
                log_path=os.path.join(parameter["TARGET_DIR"], args.path,
                                      'tf_logs', parameter["model_name"]))

img_edge_index = construct_edge_image(parameter["img_patch"])


def train_model(epoch, train_loader):
    """
        Performs one training epoch and updates the weight of the current model

        Args:
            train_loader:
            optimizer:
            epoch(int): Current epoch number

        Returns:
            None
    """
    train_loss = 0.0
    total = 0.0
    model.train()
    predict = []
    real_label = []
    for batch_idx, (img_batch, embed_batch1, org_seq, org_word_len, mask_batch1,
                    edge_cap1, gnn_mask_1, np_mask_1, labels, key_padding_mask_img) in enumerate(tqdm(train_loader)):
        embed_batch1 = {k: v.to(device) for k, v in embed_batch1.items()}
        batch = len(img_batch)
        with torch.set_grad_enabled(True):
            y, loss_contra = model(imgs=img_batch.cuda(), texts=embed_batch1, mask_batch=mask_batch1.cuda(),
                      img_edge_index=img_edge_index,
                      t1_word_seq=org_seq, txt_edge_index=edge_cap1, gnn_mask=gnn_mask_1.cuda(),
                      np_mask=np_mask_1.cuda(), img_edge_attr=None, key_padding_mask_img=key_padding_mask_img)

            loss = cross_entropy_loss(y, labels.cuda()) + loss_contra
            loss.backward()
            train_loss += float(loss.detach().item())
            optimizer.step()
            optimizer.zero_grad()  # clear gradients for this training step
        predict = predict + get_metrics(y.cpu())
        real_label = real_label + labels.cpu().numpy().tolist()
        total += batch
        torch.cuda.empty_cache()
        del img_batch, embed_batch1
    # Calculate loss and accuracy for current epoch
    logger.log(mode="train", scalar_value=train_loss / len(train_loader), epoch=epoch, scalar_name='loss')
    acc, recall, precision, f1 = get_four_metrics(real_label, predict)
    logger.log(mode="train", scalar_value=acc, epoch=epoch, scalar_name='accuracy')

    print(' Train Epoch: {} Loss: {:.4f} Acc: {:.4f} Rec: {:.4f} Pre: {:.4f} F1: {:.4f}'.format(epoch, train_loss / len(
        train_loader), acc, recall,
                                                                                                precision, f1))


def eval_validation_loss(val_loader):
    """
        Computes validation loss on the saved model, useful to resume training for an already saved model
    """
    val_loss = 0.
    predict = []
    real_label = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (img_batch, embed_batch1, org_seq, org_word_len, mask_batch1,
                        edge_cap1, gnn_mask_1, np_mask_1, labels, key_padding_mask_img) in enumerate(tqdm(val_loader)):
            embed_batch1 = {k: v.to(device) for k, v in embed_batch1.items()}
            y, loss_contra = model(imgs=img_batch.cuda(), texts=embed_batch1, mask_batch=mask_batch1.cuda(),
                      img_edge_index=img_edge_index,
                      t1_word_seq=org_seq, txt_edge_index=edge_cap1, gnn_mask=gnn_mask_1.cuda(),
                      np_mask=np_mask_1.cuda(), img_edge_attr=None, key_padding_mask_img=key_padding_mask_img)

            loss = cross_entropy_loss(y, labels.cuda())
            val_loss += float(loss.clone().detach().item())
            predict = predict + get_metrics(y.cpu())
            real_label = real_label + labels.cpu().numpy().tolist()
            torch.cuda.empty_cache()
            del img_batch, embed_batch1

        acc, recall, precision, f1 = get_four_metrics(real_label, predict)
        print(' Val Avg loss: {:.4f} Acc: {:.4f} Rec: {:.4f} Pre: {:.4f} F1: {:.4f}'.format(val_loss / len(val_loader),
                                                                                            acc, recall,
                                                                                            precision, f1))
    return val_loss


def evaluate_model(epoch, val_loader):
    """
        Performs one validation epoch and computes loss and accuracy on the validation set

        Args:
            model:
            epoch (int): Current epoch number

        Returns:
            val_loss (float): Average loss on the validation set
    """
    val_loss = 0.
    predict = []
    real_label = []
    model.eval()

    with torch.no_grad():
        for batch_idx, (img_batch, embed_batch1, org_seq, org_word_len, mask_batch1,
                        edge_cap1, gnn_mask_1, np_mask_1, labels, key_padding_mask_img) in enumerate(tqdm(val_loader)):
            embed_batch1 = {k: v.to(device) for k, v in embed_batch1.items()}
            y, loss_contra = model(imgs=img_batch.cuda(), texts=embed_batch1, mask_batch=mask_batch1.cuda(),
                      img_edge_index=img_edge_index,
                      t1_word_seq=org_seq, txt_edge_index=edge_cap1, gnn_mask=gnn_mask_1.cuda(),
                      np_mask=np_mask_1.cuda(), img_edge_attr=None, key_padding_mask_img=key_padding_mask_img)

            loss = cross_entropy_loss(y, labels.cuda())
            val_loss += float(loss.clone().detach().item())
            predict = predict + get_metrics(y.cpu())
            real_label = real_label + labels.cpu().numpy().tolist()
            torch.cuda.empty_cache()
            del img_batch, embed_batch1

        acc, recall, precision, f1 = get_four_metrics(real_label, predict)

        logger.log(mode="val", scalar_value=val_loss / len(val_loader), epoch=epoch, scalar_name='loss')
        logger.log(mode="val", scalar_value=acc, epoch=epoch, scalar_name='accuracy')
        print(' Val Epoch: {} Avg loss: {:.4f} Acc: {:.4f}  Rec: {:.4f} Pre: {:.4f} F1: {:.4f}'.format(epoch,
                                                                                                       val_loss / len(
                                                                                                           val_loader),
                                                                                                       acc, recall,
                                                                                                       precision, f1))
    return val_loss


def evaluate_model_test(epoch, test_loader):
    """
        Performs one validation epoch and computes loss and accuracy on the validation set

        Args:
            epoch (int): Current epoch number
            test_loader:

        Returns:
            val_loss (float): Average loss on the validation set
    """
    test_loss = 0.
    predict = []
    real_label = []
    model.eval()

    with torch.no_grad():
        for batch_idx, (img_batch, embed_batch1, org_seq, org_word_len, mask_batch1,
                        edge_cap1, gnn_mask_1, np_mask_1, labels, key_padding_mask_img) in enumerate(tqdm(test_loader)):
            embed_batch1 = {k: v.to(device) for k, v in embed_batch1.items()}
            with torch.set_grad_enabled(True):
                y, loss_contra = model(imgs=img_batch.cuda(), texts=embed_batch1, mask_batch=mask_batch1.cuda(),
                          img_edge_index=img_edge_index,
                          t1_word_seq=org_seq, txt_edge_index=edge_cap1, gnn_mask=gnn_mask_1.cuda(),
                          np_mask=np_mask_1.cuda(), img_edge_attr=None, key_padding_mask_img=key_padding_mask_img)

                loss = cross_entropy_loss(y, labels.cuda())
                test_loss += float(loss.clone().detach().item())
            predict = predict + get_metrics(y.cpu())
            real_label = real_label + labels.cpu().numpy().tolist()
            torch.cuda.empty_cache()
            del img_batch, embed_batch1

    acc, recall, precision, f1 = get_four_metrics(real_label, predict)

    logger.log(mode="test", scalar_value=test_loss / len(test_loader), epoch=epoch, scalar_name='loss')
    logger.log(mode="test", scalar_value=acc, epoch=epoch, scalar_name='accuracy')
    print(' Test Epoch: {} Avg loss: {:.4f} Acc: {:.4f}  Rec: {:.4f} Pre: {:.4f} F1: {:.4f}'.format(epoch,
                                                                                                    test_loss / len(
                                                                                                        test_loader),
                                                                                                    acc, recall,
                                                                                                    precision, f1))
    return test_loss


def test_match_accuracy(val_loader):
    """
    Args:
        Once the model is trained, it is used to evaluate the how accurately the captions align with the objects in the image
    """
    try:
        print("Loading Saved Model")
        checkpoint = torch.load(args.save)
        model.load_state_dict(checkpoint)
        print("Saved Model successfully loaded")
        val_loss = 0.
        predict = []
        real_label = []
        pv_list = []
        pv_know_list = []
        a_list = []
        a_know_list = []
        model.eval()
        with torch.no_grad():
            for batch_idx, (img_batch, embed_batch1, org_seq, org_word_len, mask_batch1,
                            edge_cap1, gnn_mask_1, np_mask_1, labels, key_padding_mask_img) in enumerate(tqdm(val_loader)):
                embed_batch1 = {k: v.to(device) for k, v in embed_batch1.items()}
                with torch.no_grad():
                    y, a, pv = model(imgs=img_batch.cuda(), texts=embed_batch1, mask_batch=mask_batch1.cuda(),
                                     img_edge_index=img_edge_index,
                                     t1_word_seq=org_seq, txt_edge_index=edge_cap1, gnn_mask=gnn_mask_1.cuda(),
                                     np_mask=np_mask_1.cuda(), img_edge_attr=None, key_padding_mask_img=key_padding_mask_img)

                    loss = cross_entropy_loss(y, labels.cuda())
                    val_loss += float(loss.clone().detach().item())
                predict = predict + get_metrics(y.cpu())
                real_label = real_label + labels.cpu().numpy().tolist()
                pv_list.append(pv.cpu().clone().detach())
                a_list.append(a.cpu().clone().detach())
                torch.cuda.empty_cache()
                del img_batch, embed_batch1
            acc, recall, precision, f1 = get_four_metrics(real_label, predict)
            save_result = {"real_label": real_label, 'predict_label': predict, "pv_list": pv_list,
                           " a_list": a_list}
            torch.save(save_result, "with_out_knowledge")

        print(
            "Avg loss: {:.4f} Acc: {:.4f}  Rec: {:.4f} Pre: {:.4f} F1: {:.4f}".format(val_loss, acc, recall, precision,
                                                                                      f1))
    except Exception as e:
        print(e)
        exit()


def main():
    if args.mode == 'train':
        # annotation_train = os.path.join(annotation_files, "trainknow.json")
        # annotation_val = os.path.join(annotation_files, "valknow.json")
        # annotation_test = os.path.join(annotation_files, "testknow.json")
        if knowledge_type == 0:
            annotation_train = os.path.join(annotation_files, "traindep.json")
            annotation_val = os.path.join(annotation_files, "valdep.json")
            annotation_test = os.path.join(annotation_files, "testdep.json")
        else:
            annotation_train = os.path.join(annotation_files, "trainknow_dep.json")
            annotation_val = os.path.join(annotation_files, "valknow_dep.json")
            annotation_test = os.path.join(annotation_files, "testknow_dep.json")
        img_train = os.path.join(img_files, "train_B32.pt")
        img_val = os.path.join(img_files, "val_B32.pt")
        img_test = os.path.join(img_files, "test_B32.pt")
        # 因为文件过大 训练集单独加载每个样本图像
        # img_train = os.path.join(img_files, "train_152_4.pt")
        # img_val = os.path.join(img_files, "val_152_4.pt")
        # img_test = os.path.join(img_files, "test_152_4.pt")
        train_dataset = BaseSet(type="train", max_length=parameter["max_length"], text_path=annotation_train,
                                use_np=use_np, img_path=img_train,
                                knowledge=knowledge_type)
        val_dataset = BaseSet(type="val", max_length=parameter["max_length"], text_path=annotation_val, use_np=use_np,
                              img_path=img_val, knowledge=knowledge_type)
        test_dataset = BaseSet(type="test", max_length=parameter["max_length"], text_path=annotation_test,
                               use_np=use_np, img_path=img_test, knowledge=knowledge_type)

        train_loader = DataLoader(dataset=train_dataset, batch_size=parameter["batch_size"], num_workers=4,
                                  shuffle=True,
                                  collate_fn=PadCollate_without_know())
        print("training dataset has been loaded successful!")
        val_loader = DataLoader(dataset=val_dataset, batch_size=parameter["batch_size"], num_workers=4,
                                shuffle=True,
                                collate_fn=PadCollate_without_know())
        print("validation dataset has been loaded successful!")
        test_loader = DataLoader(dataset=test_dataset, batch_size=16, num_workers=4,
                                 shuffle=True,
                                 collate_fn=PadCollate_without_know())
        print("test dataset has been loaded successful!")

        start_epoch = 0
        patience = 8

        if args.path is not None and not os.path.exists(args.path):
            os.mkdir(args.path)
        try:
            print("Loading Saved Model")
            checkpoint = torch.load(args.save)
            model.load_state_dict(checkpoint)
            start_epoch = int(re.search("-\d+", args.save).group()[1:]) + 1
            print("Saved Model successfully loaded")
            # Only effect special layers like dropout layer
            model.eval()
            best_loss = eval_validation_loss(val_loader=val_loader)
        except:
            print("Failed, No Saved Model")
            best_loss = np.Inf
        early_stop = False
        counter = 0
        for epoch in range(start_epoch + 1, parameter["epochs"] + 1):
            # Training epoch
            train_model(epoch=epoch, train_loader=train_loader)
            # Validation epoch
            avg_val_loss = evaluate_model(epoch, val_loader=val_loader)
            avg_test_loss = evaluate_model_test(epoch, test_loader=test_loader)

            scheduler.step(avg_val_loss)
            if avg_val_loss <= best_loss:
                counter = 0
                best_loss = avg_val_loss
                # torch.save(model.state_dict(), os.path.join(args.path, parameter["model_name"] + '-' + str(epoch) + '.pt'))
                print("Best model saved/updated..")
                torch.cuda.empty_cache()
            else:
                counter += 1
                if counter >= patience:
                    early_stop = True
            # If early stopping flag is true, then stop the training
            torch.save(model.state_dict(), os.path.join(args.path, parameter["model_name"] + '-' + str(epoch) + '.pt'))
            if early_stop:
                print("Early stopping")
                break

    elif args.mode == 'eval':
        # args.save
        annotation_test = os.path.join(annotation_files, "testdep.json")
        img_test = os.path.join(img_files, "test_B32.pt")

        test_dataset = BaseSet(type="test", max_length=parameter["max_length"], text_path=annotation_test,
                               use_np=use_np,
                               img_path=img_test, knowledge=parameter["knowledge_type"])
        test_loader = DataLoader(dataset=test_dataset, batch_size=parameter["batch_size"], shuffle=False,
                                 collate_fn=PadCollate(use_np=use_np, max_know_len=parameter["know_max_length"]))

        print("validation dataset has been loaded successful!")
        test_match_accuracy(val_loader=test_loader)

    else:
        print("Mode of SSGN is error!")


if __name__ == "__main__":
    main()
    # seed_everything(42)

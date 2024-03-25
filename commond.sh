#!/bin/bash
set -e

# 设置参数 默认值
mode="train"
patience=3
batch_size=32
epochs=15
max_length=120
use_np=false
seed=42
weight_decay=0.005
lr=1e-5
txt_input_dim=768
txt_out_size=200
img_input_dim=768
img_inter_dim=500
img_out_dim=200
cro_layers=6
cro_heads=5
cro_drop=0.5
txt_gat_layer=2
txt_gat_drop=0.5
txt_gat_head=2
txt_self_loops=true
img_gat_layer=2
img_gat_drop=0.5
img_gat_head=2
img_self_loops=true
img_edge_dim=0
img_patch=49
lambda=1
type_bmco=1
knowledge_type=0
know_max_length=20
know_gat_layer=2
know_gat_head=1
know_cro_layer=3
know_cro_head=5
know_cro_type=0
visualization=false

device=0
# for img_inter_dim in 200 300 400 500 
# do
# for img_gat_drop in 0 0.1 0.2 0.3 0.4
# do
for cro_layers in 2 3 5 7 8 10
do
for cro_drop in 0 0.1 0.2 0.3 0.4
do



# for 循环遍历参数

# 输出参数及其值到文件
echo "mode=$mode" > output.txt
echo "patience=$patience" >> output.txt
echo "batch_size=$batch_size" >> output.txt
echo "epochs=$epochs" >> output.txt
echo "max_length=$max_length" >> output.txt
echo "use_np=$use_np" >> output.txt
echo "seed=$seed" >> output.txt
echo "weight_decay=$weight_decay" >> output.txt
echo "lr=$lr" >> output.txt
echo "txt_input_dim=$txt_input_dim" >> output.txt
echo "txt_out_size=$txt_out_size" >> output.txt
echo "img_input_dim=$img_input_dim" >> output.txt
echo "img_inter_dim=$img_inter_dim" >> output.txt
echo "img_out_dim=$img_out_dim" >> output.txt
echo "cro_layers=$cro_layers" >> output.txt
echo "cro_heads=$cro_heads" >> output.txt
echo "cro_drop=$cro_drop" >> output.txt
echo "txt_gat_layer=$txt_gat_layer" >> output.txt
echo "txt_gat_drop=$txt_gat_drop" >> output.txt
echo "txt_gat_head=$txt_gat_head" >> output.txt
echo "txt_self_loops=$txt_self_loops" >> output.txt
echo "img_gat_layer=$img_gat_layer" >> output.txt
echo "img_gat_drop=$img_gat_drop" >> output.txt
echo "img_gat_head=$img_gat_head" >> output.txt
echo "img_self_loops=$img_self_loops" >> output.txt
echo "img_edge_dim=$img_edge_dim" >> output.txt
echo "img_patch=$img_patch" >> output.txt
echo "lambda=$lambda" >> output.txt
echo "type_bmco=$type_bmco" >> output.txt
echo "knowledge_type=$knowledge_type" >> output.txt
echo "know_max_length=$know_max_length" >> output.txt
echo "know_gat_layer=$know_gat_layer" >> output.txt
echo "know_gat_head=$know_gat_head" >> output.txt
echo "know_cro_layer=$know_cro_layer" >> output.txt
echo "know_cro_head=$know_cro_head" >> output.txt
echo "know_cro_type=$know_cro_type" >> output.txt
echo "visualization=$visualization" >> output.txt

# 运行Python脚本，并将输出追加到文件
$(CUDA_VISIBLE_DEVICES=0 python train.py \
    --patience $patience \
    --batch_size $batch_size \
    --epochs $epochs \
    --max_length $max_length \
    --use_np $use_np \
    --seed $seed \
    --weight_decay $weight_decay \
    --lr $lr \
    --txt_input_dim $txt_input_dim \
    --txt_out_size $txt_out_size \
    --img_input_dim $img_input_dim \
    --img_inter_dim $img_inter_dim \
    --img_out_dim $img_out_dim \
    --cro_layers $cro_layers \
    --cro_heads $cro_heads \
    --cro_drop $cro_drop \
    --txt_gat_layer $txt_gat_layer \
    --txt_gat_drop $txt_gat_drop \
    --txt_gat_head $txt_gat_head \
    --txt_self_loops $txt_self_loops \
    --img_gat_layer $img_gat_layer \
    --img_gat_drop $img_gat_drop \
    --img_gat_head $img_gat_head \
    --img_self_loops $img_self_loops \
    --img_edge_dim $img_edge_dim \
    --img_patch $img_patch \
    --type_bmco $type_bmco \
    --knowledge_type $knowledge_type \
    --know_max_length $know_max_length \
    --know_gat_layer $know_gat_layer \
    --know_gat_head $know_gat_head \
    --know_cro_layer $know_cro_layer \
    --know_cro_head $know_cro_head \
    --know_cro_type $know_cro_type )
    >> output.txt

done
done
# done
# done

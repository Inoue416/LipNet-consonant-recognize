import os
gpu = '0'
random_seed = 0
data_type = 'unseen'
video_path = 'data/'
train_list = 'train_dataPath.txt'
val_list = 'val_dataPath.txt'
anno_path = 'anno_data'
vid_padding = 300
txt_padding = 300#185
batch_size = 4
base_lr = 1.5e-06#2e-05
num_workers = 0
max_epoch = 50#30#100
display = 10
test_step = 100
save_prefix = f'{data_type}'
is_optimize = True

weights = 'pretrain/LipNet_unseen_loss_0.44562849402427673_wer_0.1332580699113564_cer_0.06796452465503355.pt'

log_dir=""
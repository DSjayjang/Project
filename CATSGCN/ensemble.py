import argparse
import pickle

import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', default='kinetics', choices={'kinetics', 'ntu/xsub', 'ntu/xview'},
                    help='the work folder for storing results')
parser.add_argument('--alpha', default=1, help='weighted summation')
arg = parser.parse_args()

dataset = arg.datasets
label = open('./data/fire/val_label.pkl', 'rb')
label = np.array(pickle.load(label))
r1 = open('./work_dir/' + dataset + '/agcn_joint_test/epoch1_test_score.pkl', 'rb')
r1 = list(pickle.load(r1).items())
r2 = open('./work_dir/' + dataset + '/agcn_test_bone/epoch1_test_score.pkl', 'rb')
r2 = list(pickle.load(r2).items())
right_num = total_num = 0
right_num_2 = right_num_3 = right_num_4 = right_num_5 = 0

for i in tqdm(range(len(label[0]))):
    _, l = label[:, i]
    _, r11 = r1[i]
    _, r22 = r2[i]
    r = r11 + r22 * arg.alpha
    rank_5 = r.argsort()[-5:]
    right_num_5 += int(int(l) in rank_5)
    right_num_4 += int(int(l) in rank_5[-4:])
    right_num_3 += int(int(l) in rank_5[-3:])
    right_num_2 += int(int(l) in rank_5[-2:])

    r = np.argmax(r)
    right_num += int(r == int(l))
    total_num += 1
acc1 = right_num / total_num
acc2 = right_num_2 / total_num
acc3 = right_num_3 / total_num
acc4 = right_num_4 / total_num
acc5 = right_num_5 / total_num

print(f"Top1: {acc1*100:.2f}%")
print(f"Top2: {acc2*100:.2f}%")
print(f"Top3: {acc3*100:.2f}%")
print(f"Top4: {acc4*100:.2f}%")
print(f"Top5: {acc5*100:.2f}%")
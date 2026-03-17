import os
import json
import numpy as np
from scipy.stats import bernoulli

# coordinate indices
indices = [15,16, # head
           12,13, # neck
           27,28, # right shoulder
           33,34, # right elbow
           39,40, # right wrist
           24,25, # left shoulder
           30,31, # left elbow
           36,37, # left wrist
           45,46, # right hip
           51,52, # right knee
           57,58, # right anckle
           42,43, # left hip
           48,49, # left knee
           54,55, # left anckle
           15,16, # head
           15,16, # head
           15,16, # head
           15,16] # head

# Joint index:
# {0,  "Nose"}
# {1,  "Neck"},
# {2,  "RShoulder"},
# {3,  "RElbow"},
# {4,  "RWrist"},
# {5,  "LShoulder"},
# {6,  "LElbow"},
# {7,  "LWrist"},
# {8,  "RHip"},
# {9,  "RKnee"},
# {10, "RAnkle"},
# {11, "LHip"},
# {12, "LKnee"},
# {13, "LAnkle"},
# {14, "REye"},
# {15, "LEye"},
# {16, "REar"},
# {17, "LEar"},
# # 실제 받은 json
# li=["pelvis_x","pelvis_y","pelvis_z", # 0 1
#     "spine1_x","spine1_y","spine1_z", # 3 4
#     "spine2_x","spine2_y","spine2_z", # 6 7
#     "spine3_x","spine3_y","spine3_z", # 9 10
#     "neck_x","neck_y","neck_z", # 12 13
#     "head_x","head_y","head_z", # 15 16
#     "left_clavicle_x","left_clavicle_y","left_clavicle_z", # 18 19
#     "right_clavicle_x","right_clavicle_y","right_clavicle_z", # 21 22
#     "left_shoulder_x","left_shoulder_y","left_shoulder_z", # 24 25
#     "right_shoulder_x","right_shoulder_y","right_shoulder_z", # 27 28
#     "left_elbow_x","left_elbow_y","left_elbow_z", # 30 31
#     "right_elbow_x","right_elbow_y","right_elbow_z", # 33 34
#     "left_wrist_x","left_wrist_y","left_wrist_z", # 36 37
#     "right_wrist_x","right_wrist_y","right_wrist_z", # 39 40
#     "left_hip_x","left_hip_y","left_hip_z", # 42 43
#     "right_hip_x","right_hip_y","right_hip_z", # 45 46
#     "left_knee_x","left_knee_y","left_knee_z", # 48 49
#     "right_knee_x","right_knee_y","right_knee_z", # 51 52
#     "left_ankle_x","left_ankle_y","left_ankle_z", # 54 55
#     "right_ankle_x","right_ankle_y","right_ankle_z", # 57 58
#     "left_foot_x","left_foot_y","left_foot_z", # 60 61
#     "right_foot_x","right_foot_y","right_foot_z"] # 63 64

# test vs train
te = bernoulli(0.2)

fd_name = [name for name in os.listdir('./data/fire_json/') if name != '.DS_Store']
pose_id = [name.split('_')[0] for name in fd_name if name != '.DS_Store']

for i in range(len(fd_name)):
    iact = int(pose_id[i])
    ifd = fd_name[i]
    fname = os.listdir('./data/fire_json/' + ifd)

    for j in range(len(fname)):
        ifn = fname[j]
        if ifn.split('.')[-1] == 'json':
            # print(ifn) 
            
            ### Opening JSON file
            with open(r'C:\Users\user\Desktop\연구\7. CATGCN\FIRE_raw - 복사본\fire_data\AI_DataSet_MP4_JSON\\' + ifd + '/' + ifn, 'r', encoding='utf-8') as openfile:
                # Reading from json file
                json_object = json.load(openfile)
                # print(json_object)
             
            # pose
            keypoints = json_object['annotations']['keypoints']
            nframe = len(keypoints)

            if nframe >= 300:
              dta = []
              ith_frame = 0
              for iframe in list(np.linspace(1, nframe, 300,dtype='int')):
                pose = list(np.array(keypoints[iframe-1])[indices])
                
                pose[0:len(pose):2] = list(np.array(pose[0:len(pose):2])/1920)
                pose[1:len(pose):2] = list(np.array(pose[1:len(pose):2])/1080)
                
                for i in range(0,len(pose)):
                  pose[i] = round(pose[i],3)

                # score
                score = []
                for i in range(0,18):
                  score.append(round(float(0.7 + np.random.random(1)*0.2),3))

                # Make a dictionary object
                skt = list()
                skt.append({"pose": pose, "score": score})
                dta.append({"frame_index": ith_frame, "skeleton": skt})
                ith_frame += 1

            else:  
              dta = []
              for iframe in range(1,nframe+1):
                pose = list(np.array(keypoints[iframe-1])[indices])
                
                pose[0:len(pose):2] = list(np.array(pose[0:len(pose):2])/1920)
                pose[1:len(pose):2] = list(np.array(pose[1:len(pose):2])/1080)
                
                for i in range(0,len(pose)):
                  pose[i] = round(pose[i],3)

                # score
                score = []
                for i in range(0,18):
                  score.append(round(float(0.7 + np.random.random(1)*0.2),3))

                # Make a dictionary object
                skt = list()
                skt.append({"pose": pose, "score": score})
                dta.append({"frame_index": iframe-1, "skeleton": skt})
                
              for iframe in range(nframe,300):
                # Make a dictionary object
                skt = list()
                #skt.append({"pose": [], "score": []})
                dta.append({"frame_index": iframe, "skeleton": skt})
                
            ### Writing JSON file
            out = {"data": dta,
                    "label": "a"+str(iact),
                    "label_index": iact}  
      
            tmpname = ifn.split('_')
            if te.rvs(1)[0] == 1:
              outfname = './data/fire_val/__a'+str(iact)+'_'+tmpname[1]+'_'+tmpname[2]+'_'+'f'+str(j)+'.json'
            else:
              outfname = './data/fire_train/__a'+str(iact)+'_'+tmpname[1]+'_'+tmpname[2]+'_'+'f'+str(j)+'.json'
              
            with open(outfname, "w", encoding="UTF-8") as outfile:
              json.dump(out, outfile)
              
             
    
# label
te_name = [name for name in os.listdir('./data/fire_val') if name != '.DS_Store']
val_label = {}
for i in range(0,len(te_name)):
    itr = te_name[i]
    iact = int((itr.split('_a')[1]).split('_')[0])
    print(iact)
    val_label[itr.split('.')[0]]= {
              "has_skeleton": True,
              "label": "a"+str(iact),
              "label_index": iact}

tr_name = [name for name in os.listdir('./data/fire_train') if name != '.DS_Store']
train_label = {}
for i in range(0,len(tr_name)):
    itr = tr_name[i]
    iact = int((itr.split('_a')[1]).split('_')[0])
    print(iact)
    train_label[itr.split('.')[0]]= {
                "has_skeleton": True,
                "label": "a"+str(iact),
                "label_index": iact}
            
with open('./data/fire_val_label.json','a', encoding="UTF-8") as outfile2:
  json.dump(val_label, outfile2, indent = 4)

with open('./data/fire_train_label.json','a', encoding="UTF-8") as outfile3:
  json.dump(train_label, outfile3, indent = 4)


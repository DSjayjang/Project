import os
import json
import numpy as np
from scipy.stats import bernoulli

# coordinate indices
indices = [15,16,12,13,18,19,24,25,30,31,36,37,21,22,27,28,33,34,39,40,0,1,6,7,42,43,48,49,60,61,45,46,51,52,63,64]

# test vs train
te = bernoulli(0.2)

fd_name = [name for name in os.listdir('./data/fire_json/') if name != '.DS_Store']
pose_id = [name.split('_')[0] for name in fd_name if name != '.DS_Store']

for i in range(0,len(fd_name)):
    iact = int(pose_id[i])
    ifd = fd_name[i]
    fname = os.listdir('./data/fire_json/'+ifd)
    for j in range(0,len(fname)):
        ifn = fname[j]
        if ifn.split('.')[-1] == 'json':
           print(ifn) 
           
           ### Opening JSON file
           with open('./data/fire_json/'+ifd+'/'+ifn, 'r', encoding='utf-8') as openfile:
             # Reading from json file
             json_object = json.load(openfile)
             #print(json_object)
             
           # pose
           keypoints = json_object['annotations']['keypoints']
           nframe = len(keypoints)
           if nframe >= 300:
             dta = []
             ith_frame = 1
             for iframe in list(np.linspace(1,nframe,299,dtype='int')):
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
               dta.append({"frame_index": iframe, "skeleton": skt})
               
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


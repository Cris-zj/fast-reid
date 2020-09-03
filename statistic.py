import os
import sys

# Statistic all images
txtpath = '/home/lijiaqi/LiuHongzhi/ReID/fast-reid/datasets/NAIC/train/label.txt'
cnt_dict = {}
f = open(txtpath,"r")
for line in f.readlines():
    line = line.strip('\n')
    # print("line: ",line)
    img, pid = line.strip('\n').split(':')
    if pid in cnt_dict:
        cnt_dict[pid] += 1
    else:
        cnt_dict[pid] = 1
    # print(img, " ", pid)
f.close()
print(cnt_dict)
file = open("NAIC images", "w")
count = 0
all_dict = sorted(cnt_dict.items(), key=lambda x:x[1])
print(all_dict)
# wfile = open(r'more.txt','w')
wfile = open(r'less.txt','w')
for item in all_dict:
    print(item)
    # print("PID: " + str(item[0])+ "\tNum: " + str(item[1]),file=wfile)
    if item[1] < 4: 
        # continue
        print(item[0], file=wfile)
    else:
        # print(item[0], file=wfile)
        continue
wfile.close()


# label_file = '/home/lijiaqi/LiuHongzhi/ReID/fast-reid/datasets/NAIC/train/label.txt'
# need_file = '/home/lijiaqi/LiuHongzhi/ReID/fast-reid/more.txt'
# new_file = '/home/lijiaqi/LiuHongzhi/ReID/fast-reid/datasets/NAIC/train/new_label.txt'

# need_id = []
# f = open(need_file,"r")
# for line in f.readlines():
#     line = line.strip()
#     print(line)
#     need_id.append(line)
# print("need id length: ",need_id)
# f.close()

# rfile = open(label_file,"r")
# wfile = open(new_file,"w")
# for l in rfile.readlines():
#     l = l.strip('\n')
#     print(l)
#     wl = l.split(":")[1]
#     print(wl)
#     if wl in need_id:
#         print(l, file=wfile)
# wfile.close()
# rfile.close()





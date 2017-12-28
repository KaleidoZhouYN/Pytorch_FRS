caffe_root = '/home/zhouyn/caffe/python' # change your pycaffe root here
import sys
sys.path.append(caffe_root)

import caffe
import numpy as np

model = '/data2/zhouyinan/exp/model/softmax-28L-128/face_concat_iter_100000.caffemodel'   #change your .caffemodel file here
proto = '/home/zhouyn/FaceRecognitionScript/softmax-28L-128/face_model_concat.prototxt'   #change your .prototxt file here

faceNet = caffe.Net(proto,model,caffe.TEST)

fc6_1 = faceNet.params['fc6_1'][0].data  # classification weights of MS
fc6_2 = faceNet.params['fc6_2'][0].data  # classification weights of VGG2

out = np.dot(fc6_2,fc6_1.T)   

f = open('./reduplicton_id.txt','w')

for i in range(out.shape[0]):
    if (i % 1000 == 0):
        print(i)
    for j in range(out.shape[1]):
        if (out[i,j]>0.5): #0.5 is threshold
            print(str(i)+' '+str(j)+' '+ str(out[i,j]))
            f.write(str(i)+' '+str(j)+' '+ str(out[i,j]) + '\n')

f.close()




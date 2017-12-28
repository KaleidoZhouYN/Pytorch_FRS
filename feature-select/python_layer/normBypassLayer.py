import sys
sys.path.append('/home/zhouyn/caffe/python')
import caffe

import numpy as np

class NormBypassLayer(caffe.Layer):
    '''
        this class aim to automatically choose low norm feature to form a label 
    '''
    def setup(self,bottom,top):
        '''
            param:
                norm:threshold norm[default:0] -1:avgnorm
                output:feature dim[neccessary]
        '''
        #config
        self.norm_ = 0
        params = eval(self.param_str)
        if ('norm' in params.keys()):
            self.norm_ = params['norm']
        self.output_ = int(params['output'])
            
        self.avgnorm_ = 0
        
        #two tops:
        #if len(top) != 2:
        #    raise Exception("top number must be two")
            
        if len(bottom) != 1:
            raise Exception("bottom number must be one")
            
    def reshape(self,bottom,top):
        '''
            bottom[0]: fc5(feature layer) shape:NC
            top[0]:fc5(feature layer) shape:NC
            top[1]:label shape:N
        '''
        #print(bottom[0].data.shape)
        top[0].reshape(bottom[0].data.shape[0],self.output_)
        if (len(top) == 2):
            top[1].reshape(bottom[0].data.shape[0])
    
    def forward(self,bottom,top):
        '''
            copy bottom[0] to top[0]
            judge norm of bottom[0] and create lable
        '''
        #print('forward')
        np.copyto(top[0].data,np.zeros(top[0].data.shape))
        
        tmp_avg = 0
        if (self.norm_ == -1):
            avg_norm = self.avgnorm_
        else:
            avg_norm = self.norm_
        for n in range(bottom[0].data.shape[0]):
            norm = np.linalg.norm(bottom[0].data[n])
            if (norm < avg_norm):
                if(len(top) == 2):
                    top[1].data[n] = 1
            else:
                if (len(top) == 2):
                    top[1].data[n] = 0
                np.copyto(top[0].data[n,0:bottom[0].data.shape[1]],bottom[0].data[n])
            tmp_avg += norm
        tmp_avg /= bottom[0].data.shape[0]
        if (self.avgnorm_ == 0):
            self.avgnorm_ = tmp_avg
        else:
            self.avgnorm_ = (self.avgnorm_ + tmp_avg) / 2        
        
    def backward(self,top,propagate_down,bottom):
        #print('backward')
        if (self.norm_ == -1):
            avg_norm = self.avgnorm_
        else:
            avg_norm = self.norm_        
        for n in range(bottom[0].data.shape[0]):
            norm = np.linalg.norm(bottom[0].data[n])
            if (norm > avg_norm):            
                np.copyto(bottom[0].diff[n],top[0].diff[n,0:bottom[0].data.shape[1]])
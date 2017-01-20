# Formula : output_size = (input_size - kernel_size + pad*2)/stride + 1
# (out - 1)*stride + kernel - pad*2 = input
# Each Layer : (kernel_size, stride, pad)
from parse_caffe_pt import parse_pt
import pdb
# AlexNet
#layer_params =[[11,4,0],[3,2,0],[5,1,2],[3,2,0],[3,1,1],[3,1,1],[3,1,1],[3,2,0]]
#layer_names = ['conv1','pool1','conv2','pool2','conv3','conv4','conv5','pool5']
layer_names, layer_params = parse_pt('VGG_ILSVRC_16_layers_deploy.prototxt')
#layer_names, layer_params = parse_pt('deploy.prototxt')



def backward(output_dim, params):
    #kernel = params[0]
    #stride = params[1]
    #pad = params[2]
    kernel = params['kernel_size']
    stride = params['stride']
    pad = params['pad']
    input_size = (output_dim - 1)*stride + kernel #- pad*2
    
    return input_size

def forward(input_dim, params):
    kernel = params['kernel_size']
    stride = params['stride']
    pad = params['pad']
    output_size = (input_dim - kernel )/stride + 1
    return output_size

if __name__ == '__main__':
    output_dim = 1
    
    for i in xrange(len(layer_names)):
        idx = len(layer_names)-i-1
        print layer_names[idx]
        output_dim = backward(output_dim,layer_params[idx])
        print output_dim
    '''
    input_dim=195
    for i in xrange(len(layer_names)):
        idx = i
        input_dim = forward(input_dim,layer_params[idx])
    print input_dim
   '''

import pdb


def parse_pt(filename):
    f=open(filename)
    params = []
    #kernel_size, stride, pad
    names = []
    layer_req=False
    while True:
        la=f.readline().strip()
        if 'layer' in la and len(la) <= 8:
            if layer_req == True:
                names.append(name)
                params.append(pz)
            pz = {'kernel_size':0, 'stride':1, 'pad':0}
            layer_req=False
        if 'type' in la :
            if 'CONVOLUTION' in la or 'POOLING' in la:
                layer_req=True
        if 'kernel_size' in la:
            pz['kernel_size'] = int(la.split()[-1])
        elif 'stride' in la:
            pz['stride'] = int(la.split()[-1])
        elif 'pad' in la:
            pz['pad'] = int(la.split()[-1])
        elif 'name' in la:
            name = la.split()[-1]
        if not la:
            break
    return names, params




if __name__ == '__main__':
    n,p=parse_pt('VGG_ILSVRC_16_layers_deploy.prototxt')
    pdb.set_trace()
    #n,p=parse_pt('deploy.prototxt')
    for i in xrange(len(n)):
        print n[i],p[i]
    pdb.set_trace()

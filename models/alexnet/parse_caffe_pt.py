import pdb


def parse_pt(filename):
    f=open(filename)
    params = []
    #kernel_size, stride, pad
    names = []
    layer_req=False
    while True:
        l=f.readline().strip()
        if 'layer' in l and len(l) < 8:
            if layer_req == True:
                names.append(name)
                params.append(p)
            p = {'kernel_size':0, 'stride':1, 'pad':0}
            layer_req=False
        if 'type' in l :
            if 'Convolution' in l or 'Pooling' in l:
                layer_req=True
        if 'kernel_size' in l:
            p['kernel_size'] = int(l.split()[-1])
        elif 'stride' in l:
            p['stride'] = int(l.split()[-1])
        elif 'pad' in l:
            p['pad'] = int(l.split()[-1])
        elif 'name' in l:
            name = l.split()[-1]
        if not l:
            break
    return names, params




if __name__ == '__main__':
    #n,p=parse_pt('VGG_ILSVRC_16_layers_deploy.prototxt')
    n,p=parse_pt('deploy.prototxt')
    for i in xrange(len(n)):
        print n[i],p[i]
    pdb.set_trace()

import pdb
def loadfile(filename):
    with open(filename, 'r') as infile:
        line = ''
        iter_dict = {}
        accuracy = {}
        test_loss = {}
        while True:
            print line
            if 'Test net output #0' in line:
                accuracy[iteration] = float(line.split(' ')[14].split('\n')[0])
            if 'Test net output #1' in line:
                test_loss[iteration] = float(line.split(' ')[14].split('\n')[0])

            if 'Iteration' not in line:
                try:
                    line = next(infile)
                    continue
                except:
                    return iter_dict,accuracy,test_loss
            entry = {}
            set_index = 5
            iteration = line.split(' ')[set_index].split(',')[0] #5-->6
            string = line.split(' ')[set_index+1] #6-->7
            print iteration
            print string
            try:
                if (int(iteration) % 5000 == 0) and (string == 'loss'):
                    loss = line.split(' ')[set_index+3].split('\n')[0] #8 --> 9
                    loss = float(loss)
                    print loss
                    #if loss > 3.0:
                    #    loss = 3.0
                    iter_dict[iteration] = loss
            except ValueError:
                print 'value error'
            line = next(infile)

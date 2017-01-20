import parseLoss as pl
import pdb
import matplotlib.pyplot as plt
import matplotlib as mt

#mt.use('Agg')
#plt.ioff()
logname = 'train_val_lr0001' # remember to change this
path = '../logfile/%s.log' % logname
loss, accuracy, test_loss = pl.loadfile(path)
fig1 = plt.figure()
plt.bar(range(len(loss)), loss.values(), align='center')
plt.xticks(range(len(loss)), loss.keys())
plt.ylabel('loss')
plt.xlabel('iteration')
fig1.savefig('loss_%s.png'% logname)
#fig.savefig('~/Dropbox/schoolprint/lab/meeting/meeting19092016/loss_lrdot05.png')
fig2 = plt.figure()
plt.bar(range(len(accuracy)), accuracy.values(), align='center')
plt.xticks(range(len(accuracy)), accuracy.keys())
plt.ylabel('accuracy')
plt.xlabel('iteration')
fig2.savefig('accu_%s.png'% logname)

fig3 = plt.figure()
plt.bar(range(len(test_loss)), test_loss.values(), align='center')
plt.xticks(range(len(test_loss)), test_loss.keys())
plt.ylabel('test_loss')
plt.xlabel('iteration')
fig3.savefig('test_loss_%s.png'% logname)

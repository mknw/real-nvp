#!/var/scratch/mao540/miniconda3/envs/maip-venv/bin/python3.7

from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
import pandas


def plot_loss(filename, log_fp='data/res_3-8-32/log', train=False, test=True, hi_epochs=None,
                  ylimits=(-10, 1100), threshold=1100, loss=True, bpd=True):

    if not hi_epochs:
        hi_epochs=[121, 133, 138, 160, 164, 182, 196, 240, 251, 252, 254]

    df = pandas.read_csv(log_fp, names=['epoch', 'train_loss', 'train_bpd', 'test_loss', 'test_bpd'])
    plot_title = ''

    if threshold:
        df[df>threshold] = None


    fig, ax = plt.subplots()
    if ylimits:
        plt.ylim(ylimits)

    if train:
        train_alpha = 0.35 if test else 1
        if loss:
            df.plot(kind='line', x='epoch', y='train_loss', color='#cb4b16', alpha=train_alpha, ax=ax) # solarized orange.
        if bpd:
            df.plot(kind='line', x='epoch', y='train_bpd', color='#2aa198', alpha=train_alpha, ax=ax) # solarized cyan.
    if test:
        if loss:
            df.plot(kind='line', x='epoch', y='test_loss', color='#dc322f', ax=ax) # solarized red.
        if bpd:
            df.plot(kind='line', x='epoch', y='test_bpd', color='#268bd2', ax=ax) # solarized blue.

    for epoch in hi_epochs:
        if epoch == hi_epochs[0]:      #'#859900'
            plt.axvline(x=epoch, ymin=0, c='black', ymax=ylimits[1], label='data analysed', alpha=0.06) # solarized green.
        else:
            plt.axvline(x=epoch, ymin=0, c='black', ymax=ylimits[1], alpha=0.06) # solarized green.

    

    stats_legend = plt.legend()
    # ax = plt.gca().add_artist(stats_legend)
    print(stats_legend)
    
    # analysed_epochs_patch = mpatches.Patch(color='#859900', label='analysed epochs')

    plt.title("Learning stats")
    plt.savefig(filename)


if __name__ == '__main__':
    # with everything
    param_dic = {'filename': './figs/test_train_bpdloss_trimmed_1100.png',
                     'threshold': None,
                             'train': True,
                             'test': True,
                             'loss': True,
                             'bpd': True
                             }
                             
    plot_loss(**param_dic)



#!/var/scratch/mao540/miniconda3/envs/maip-venv/bin/python3.7

from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
import pandas


def plot_loss(filename, log_fp='data/res_3-8-32/log', train=False, test=True, hi_epochs=None,
                  ylimits=None, threshold=None, loss=True, bpd=True):

    df = pandas.read_csv(log_fp, names=['epoch', 'test_loss', 'test_bpd', 'train_loss', 'train_bpd'])
    # plot_title = ''

    if threshold:
        df[df>threshold] = None


    fig, ax = plt.subplots(figsize=(10, 7))
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
            df.plot(kind='line', x='epoch', y='test_loss', color='#dc322f', linewidth=1, ax=ax) # solarized red.
        if bpd:
            df.plot(kind='line', x='epoch', y='test_bpd', color='#268bd2', linewidth=1, ax=ax) # solarized blue.
    
    if hi_epochs:
        for epoch in hi_epochs:
            # dirty, sexy fix for multiple legends.
            if epoch == hi_epochs[0]:      #'#859900'
                plt.axvline(x=epoch, ymin=0, c='black', ymax=ylimits[1], label='analysed', alpha=0.1) # solarized green.
            else:
                plt.axvline(x=epoch, ymin=0, c='black', ymax=ylimits[1], alpha=0.1) # solarized green.

    plt.tight_layout()

    stats_legend = plt.legend()
    ax = plt.gca().add_artist(stats_legend)
    print(stats_legend)
    
    # analysed_epochs_patch = mpatches.Patch(color='#859900', label='analysed epochs')

    plt.title("Learning stats")
    plt.savefig(filename)

# MNIST params #
den_mnist_param_loss= {
                 'log_fp' : 'data/dense_test6/log',
                 'filename': './figs/dmnist_loss.png',
                 # 'ylimits': (400, 1100),
                 'ylimits': (600, 2500),
                 # 'threshold': 1100,
                         'train': True,
                         'test': True,
                         'loss': True,
                         'bpd': False,
                         'hi_epochs': range(120, 690, 10)
                         }

den_mnist_param_bpd = {
                 'log_fp' : 'data/dense_test6/log',
                 'filename': './figs/dmnist_bpd.png',
                 'ylimits': (0, 5),
                 # 'threshold': 1100,
                         'train': True,
                         'test': True,
                         'loss': False,
                         'bpd': True,
                         'hi_epochs': 0 # [121, 133, 138, 160, 164, 182, 196, 240, 251, 252, 254]
                         }

res_mnist_param_loss = {
                 'log_fp' : 'data/res_3-8-32/log',
                 'filename': './figs/rmnist_loss.png',
                 # 'ylimits': (400, 1100),
                 'ylimits': (600, 3300),
                 # 'threshold': 1100,
                         'train': True,
                         'test': True,
                         'loss': True,
                         'bpd': False,
                         'hi_epochs': 0 # [121, 133, 138, 160, 164, 182, 196, 240, 251, 252, 254]
                         }

res_mnist_param_bpd = {
                 'log_fp' : 'data/res_3-8-32/log',
                 'filename': './figs/rmnist_bpd.png',
                 'ylimits': (0, 6),
                 # 'threshold': 1100,
                         'train': True,
                         'test': True,
                         'loss': False,
                         'bpd': True,
                         'hi_epochs': 0 # [121, 133, 138, 160, 164, 182, 196, 240, 251, 252, 254]
                         }

# CelebA params #
den_celeba_params_loss = {
                                             'log_fp': 'data/1_den_celeba/log',
                                             'filename': './figs/dceleba_loss.png',
                                             'train': True,
                                             'test': True,
                                             'ylimits': (20000, 28000),
                                             'loss': True,
                                             'bpd': False
                                             }

den_celeba_params_bpd = {
                                            'log_fp': 'data/1_den_celeba/log',
                                            'filename': './figs/dceleba_bpd.png',
                                            'train': True,
                                            'test': True,
                                            'ylimits': (2.5, 3.3),
                                            'loss': False,
                                            'bpd': True
                                            }



if __name__ == '__main__':
    # with everything

    # plot_loss(**den_mnist_param_loss)
    # plot_loss(**den_mnist_param_bpd)
    plot_loss(**res_mnist_param_loss)
    plot_loss(**res_mnist_param_bpd)
    plot_loss(**den_celeba_params_loss)
    plot_loss(**den_celeba_params_bpd)






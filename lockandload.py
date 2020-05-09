#!/var/scratch/mao540/miniconda3/envs/maip-venv/bin/python

import torch 
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms

import util
import argparse
import os
import numpy as np
from models import RealNVP, RealNVPLoss
from tqdm import tqdm
from random import randrange

import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from umap import UMAP


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() and len(args.gpu_ids) > 0 else "cpu")
    print("evaluating on: %s" % device)

    # select model.
    fp_model_root, fp_model, fp_vmarker = select_model(args.root_dir, args.version) # , test=440)

    mark_version(args.version, fp_vmarker) # echo '\nV-' >> fp_vmarker

    epoch = fp_model_root.split('_')[-1]
    ''' stats filenames memo: '''
    # 1: '/z_mean_std.pkl'
    # 2: '/latent_mean_std_Z.pkl'
    stats_filename = fp_model_root + '/zlatent_mean_std_Z.pkl'
    if os.path.isfile(stats_filename) and not args.force:
        print('Found cached file, skipping computations of mean and std.')
        stats = torch.load(stats_filename)
    else:
        if args.dataset == 'MNIST':
            transform_train = transforms.Compose([
                transforms.ToTensor()
                # transforms.ColorJitter(brightness=0.3)
            ])
            #torchvision.transforms.Normalize((0.1307,), (0.3081,)) # mean, std, inplace=False.
            transform_test = transforms.Compose([
                transforms.ToTensor()
            ])
            # trainset = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transform_train)
            # trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            testset = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transform_test)
            testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            print('Building model..') 

        elif args.dataset == 'CIFAR-10':
            # Note: No normalization applied, since RealNVP expects inputs in (0, 1).
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
            #torchvision.transforms.Normalize((0.1307,), (0.3081,)) # mean, std, inplace=False.
            transform_test = transforms.Compose([
                transforms.ToTensor()
            ])
            trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)
            trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)
            testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            print('Building model..')
            net = RealNVP(num_scales=2, in_channels=3, mid_channels=64, num_blocks=8)

        net = load_network( fp_model, device, args)
        loss_fn = RealNVPLoss()
        # stats = track_distribution(net, testloader, device, loss_fn)
        stats = track_z(net, testloader, device, loss_fn)
        torch.save(stats, stats_filename)

    ''' filepaths '''
    fp_distr = fp_model_root + '/distr' # distributions
    fp_simil = fp_model_root + '/similarity' # similarity
    fp_replace = fp_model_root+'/replace_from_grandz'
    fp_pca = fp_model_root + '/pca'
    [os.makedirs(ppp, exist_ok=True) for ppp in [fp_distr, fp_simil, fp_replace, fp_pca]]
    
    
    ''' distributions analyses '''
    print("analysing z distribution... ")
    scatter_alldigits(stats, fp_distr + '/meanstds.png', epoch)
    scatter_eachdigit(stats, fp_distr + '/dig_subplots.png', epoch)
    violin_eachdigit(stats, fp_distr + '/dig_violins.png', epoch)
    ''' distance analysis '''
    distances = calculate_distance(stats, joint=True)
    heatmap(distances, fp_simil + '/distances.png')
    distances = calculate_distance(stats, measure='mean')
    heatmap(distances, fp_simil + '/distances_mean.png',
                plot_title='Average distance between digits in Z space (means)')
    distances = calculate_distance(stats, measure='std')
    heatmap(distances, fp_simil + '/distances_std.png',
                plot_title='Average distance between digits in Z space (std)')

    distances = y_distance_z(stats)
    measure = 'mean'
    for m in range(distances.shape[0]):
        heatmap(distances[m], fp_simil + f'/pixelwise_dist_{measure}.png',
                                 plot_title=f'pixelwise {measure} similarity')
        measure = 'std'

    all_nz = grand_z(stats)
    if 'net' not in dir():
        net = load_network( fp_model, device, args)
    # z=craft_z(all_nz,kept=3) #helper function, included in sample_from_crafted_z.
    for k in [5, 10, 25, 50, 100, 200]:
        sample_from_crafted_z(net, all_nz, absolute=True, kept=k, device=device, reps=10,
                                     save_dir=fp_replace) #monster_mode=True)
    # overall Z for each digit (averaged across batches).
    plot_grand_z(all_nz, fp_model_root + '/grand_zs.png')
    

    ''' dimensionality reduction '''
    # PCA
    for method in ['sklearn']: # 'self']:
        PCs = PCA_test(stats['z'], k=5, center=True, mode=method)
        fn = fp_pca + '/pca_{}.png'.format(method)
        plot_PCA(PCs, filename=fn)
        fn = fp_pca + '/pcaVE_{}.png'.format(method)
        plot_expvar(PCs['exp_var'], fn)
        fn = fp_pca + '/PCgrid_{}.png'.format(method)
        plot_PCgrid(PCs, fn)
        fn = fp_pca + '/reduced_z.png'
        plot_reconstructed_PCA(PCs, fn)

    # UMAP -- `test_umap` will use directories `<fp_model_root>/umap/{,3d}`
    fn_prefix = fp_model_root + '/umap'
    os.makedirs(fn_prefix+'/3d', exist_ok=True)

    for nn in [7, 10, 20]:
        # add option to reduce nn for 3d. 
        dims = [2, 3] if (nn % 10 == 0) else [2]
        for md in range(2, 8, 2):
            for d in dims:
                test_umap(stats, fn_prefix, n_neighbors=nn, min_dist=md*0.05, n_components=d)

    mark_version(args.version, fp_vmarker, finish=True) # echo '0.2' >> fp_vmarker

    return


def issue_z_from_pc(PC, stats, filename):
    raise NotImplementedError
    z_s = grand_z(stats)
    z_s = z_s.reshape(z_s.shape[0], -1)
    components = PC['components']
    for i in range(10):
        
        print('h'+i)
        # every N
        for c in components:
            # every PC
            pass
    pass

def label_zs(z_s):
    ''' Arg:
        z_s list of ndarrays '''
    change_index = [z.shape[0] for z in z_s] # + 1 digit [0-9]
    # Prepare dataset:
    n_datapoints = np.sum(change_index)
    assert n_datapoints == 10000
    z = np.concatenate(z_s).reshape(n_datapoints, -1)
    # track target category:
    fill_val = 0
    y = np.array([]).astype(int)
    for ci in change_index:
        arr = np.array([fill_val]).repeat(ci)
        y = np.concatenate([y, arr])
        fill_val += 1
    return z, y

def test_umap(stats, fn_prefix, n_neighbors=15, min_dist=0.1, n_components=2,
                  metric='euclidean', **kwargs):

    dataset, col_arr = label_zs(stats['z'])

    print('computing UMAP projection: ', end='')
    print(f'n_neighbors = {n_neighbors}; min_dist = {min_dist}...', end='')

    reductor = UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        n_components=n_components, 
                                    metric=metric, random_state=42)
    embeddings = reductor.fit_transform(dataset)

    if n_components == 2:
        fig, ax = plt.subplots(figsize=(12,12))
        scatter = ax.scatter(embeddings[:,0], embeddings[:,1], c=col_arr, cmap='Spectral', s=4)
    elif n_components == 3:
        fn_prefix += '/3d'
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(11, 11))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(embeddings[:,0], embeddings[:,1], embeddings[:,2], alpha=0.3,
                                                c=col_arr, cmap='Spectral', s=4)
    
    handles, labels = scatter.legend_elements(prop='colors')
    ax.legend(handles, labels, loc='best', title='off-line digits')
    plt.title(f'n_neighbors = {n_neighbors}; min_dist = {min_dist}')

    filename = fn_prefix + '/nn{:d}_md{:.1f}.jpg'.format(n_neighbors, min_dist, n_components)
    plt.savefig(filename)
    plt.close()
    print(' Saved {}'.format(filename.split('/')[-1]))


def plot_PCgrid(PCs, filename, reconstruct=False):
    components = PCs['components']
    var_exp = PCs['exp_var']
    z_s = PCs['z']
    # y = PCs['y']
    if reconstruct:
        reduced_z = components.dot(z_s.T)
    components = torch.from_numpy(components.reshape(PCs['k'], 1, 28, 28)).float().to('cpu')
    PC_grid = torchvision.utils.make_grid(components, nrow=int(PCs['k'] ** 0.5), pad_value=100)
    torchvision.utils.save_image(PC_grid, filename)
    del components, var_exp, z_s


def PCA_eig_np(Z, k, center=True, scale=False):
    '''
    https://medium.com/@ravikalia/pca-done-from-scratch-with-python-2b5eb2790bfc
    '''
    n, p = Z.shape
    ones = np.ones([n, 1])
    # subtract from each column its mean, to ensure mean = 0.
    h = ((1/n) * np.matmul(ones, ones.T)) if center else np.zeros([n, n])
    H = np.eye(n) - h
    Z_center = np.matmul(H, Z)
    covariance = 1/(n-1) * np.matmul(Z_center.T, Z_center)
    # divide each column by its std. Only if outcome is independent of variance.
    scaling = np.sqrt(1/np.diag(covariance)) if scale else np.ones(p)
    scaled_covariance = np.matmul(np.diag(scaling), covariance)
    w, v = np.linalg.eig(scaled_covariance)
    components = v[:, :k]
    explained_variance = w[:k]
    return {'z': Z, 'k': k, 'components': components.T,
                'exp_var': explained_variance}

def PCA_test(z_s, k, center=False, mode='self'):
    
    dataset, y = label_zs(z_s)
    n_datapoints = dataset.shape[0]
    p = dataset.shape[1] # feature number

    if mode == 'sklearn':
        PCs = PCA(n_components=k).fit(dataset) # dataset.data
        return {'z': dataset, 'k': k, 'components': PCs.components_,
                    'exp_var': PCs.explained_variance_, 'y': y}
    elif mode == 'self':
        PCs = PCA_eig_np(dataset, k, center)
        PCs['y'] = y
        return PCs


def plot_reconstructed_PCA(PCs, filename):
    # sort from highest variance
    if isinstance(PCs, dict):
        components = PCs['components'][::-1]
        var_exp = PCs['exp_var'][::-1]
        z_s = PCs['z']
        y = PCs['y']
        reduced_z = components.dot(z_s.T) # might want to check this.
    else: # should be type: sklearn.decomposition.PCA
        raise NotImplementedError
        components = PCs.components_[::-1]
        var_exp = PCs.explained_variance_[::-1]
        reduced_Z = PCs.transform()

    n_pcs = len(var_exp) # can use this to index components and create grid.
    fig, axs = plt.subplots(n_pcs, n_pcs, figsize=(12, 12), sharex='col', sharey='row')
    
    for row in range(n_pcs):
        for col in range(n_pcs):
            if row > col:
                axs[row, col].scatter(reduced_z[row], reduced_z[col], c=y, s=.50, alpha=0.6)
                axs[row, col].annotate('var.exp.:\nC{}={:.3f}\nC{}={:.3f}'.format(n_pcs - row, var_exp[row],
                                         n_pcs-col, var_exp[col]), xy=(7, 7), fontsize='xx-small')
                if row == n_pcs-1:
                    axs[row, col].set_xlabel(f'component {n_pcs-col}') 
                    axs[row, col].tick_params(axis='x', reset=True, labelsize='x-small')
                if col == 0:
                    axs[row, col].set_ylabel(f'component {n_pcs-row}')
                    axs[row, col].tick_params(axis='y', reset=True, labelsize='x-small')
            else:
                axs[row, col].remove()
                axs[row, col] = None
    
    fig.tight_layout()
    fig.subplots_adjust(top=.88)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def plot_PCA(PCs, filename):
    # sort from highest variance
    if isinstance(PCs, dict):
        components = PCs['components'][::-1]
        var_exp = PCs['exp_var'][::-1]
        z_s = PCs['z']
        y = PCs['y']
        reduced_z = components.dot(z_s.T) # might want to check this.
    else: # should be type: sklearn.decomposition.PCA
        components = PCs.components_[::-1]
        var_exp = PCs.explained_variance_[::-1]
        reduced_Z = PCs.transform()

    n_pcs = len(var_exp) # can use this to index components and create grid.
    fig, axs = plt.subplots(n_pcs, n_pcs, figsize=(12, 12), sharex='col', sharey='row')
    
    for row in range(n_pcs):
        for col in range(n_pcs):
            if row > col:
                axs[row, col].scatter(components[row], components[col], s=.50)# label=f'{col}x{row}')
                axs[row, col].annotate('var.exp.:\nC{}={:.3f}\nC{}={:.3f}'.format(n_pcs - row, var_exp[row],
                                         n_pcs-col, var_exp[col]), xy=(.30, .30), fontsize='xx-small')
                if row == n_pcs-1:
                    axs[row, col].set_xlabel(f'component {n_pcs-col}') 
                    axs[row, col].tick_params(axis='x', reset=True, labelsize='x-small', which='both')
                if col == 0:
                    axs[row, col].set_ylabel(f'component {n_pcs-row}')
                    axs[row, col].tick_params(axis='y', reset=True, labelsize='x-small', which='both')
            else:
                axs[row, col].remove()
                axs[row, col] = None
    
    fig.tight_layout()
    fig.subplots_adjust(top=.88)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

#	horrible i know
#	nrows = ncols = len(var_exp) # can use this to index components and create grid.
#	fig, axs = plt.subplots(nrows, ncols, figsize=(12, 12), sharex='col', sharey='row')
#	
#	for row in range(nrows):
#		for col in range(ncols):
#			if row > col:
#				axs[row, col].scatter(components[row], components[col], s=.50)# label=f'{col}x{row}')
#				axs[row, col].annotate('var.exp.:\nC{}={:.3f}\nC{}={:.3f}'.format(5 - row, var_exp[row],
#					                     5-col, var_exp[col]), xy=(.30, .30), fontsize='xx-small')
#				if row == nrows-1:
#					axs[row, col].set_xlabel(f'component {5-col}') 
#				if col == 0:
#					axs[row, col].set_ylabel(f'component {5-row}')
#				if row == nrows-1 or col == 0:
#					axs[row, col].tick_params(reset=True, labelsize='x-small')
#			else:
#				axs[row, col].remove()
#				axs[row, col] = None
    

def plot_expvar(var_exp, filename):
    
    fig, ax = plt.subplots()

    ax.plot(range(len(var_exp)), var_exp)
    plt.title("variance explained by components")
    plt.savefig(filename)


def test_arrays():
    arr_1 = np.arange(24).reshape(2,3,4)
    arr1 = arr_1[0,::2].copy()
    arr2 = arr_1[1,1:].copy()
    arr_1[0,::2] = arr2[:,::-1]
    arr_1[1,1:] = arr1[:,::-1]
    arr_3= np.zeros(shape=(arr_1.shape[0], 1, 3, 4))
    arr_2 = np.zeros(shape=(arr_1.shape[0], 3, 4))
    arr_2[:] = 99
    return (arr_1, arr_2, arr_3)

def replace_highest_along_axis(arr_ref, arr_2, arr_3, k):
    '''
    Replaces values from arr_2 to arr_3, according to highest k values
    in arr_ref.
    Input:
        - arr_ref: reference array to look up highest k values.
        - arr_2: source array, to extract values from.
        - arr_3: destination array, to inject values at given indices.
    Output:
        - arr_3 with vals from arr_2 at indices presenting arr_1 k-highest values. 
    '''
    assert (arr_ref.shape[0] % arr_2.shape[0]) == 0, "Arrays mismatch"
    # arr_ref, arr_2, arr_3 = test_arrays()
    shape_arr_3 = arr_3.shape
    shape_arr_2 = arr_2.shape
    arr_mask = np.zeros_like(arr_2)
    reps = arr_3.shape[0] // arr_2.shape[0] # repetitions.
    # Reference array, with flattened batches.
    arr_ref_fb = arr_ref.reshape(arr_ref.shape[0], -1)
    # select k highest values. 
    maxk_b_ind = np.argpartition(arr_ref_fb, -k)[:, -k:] # without first :?
    maxk = np.take_along_axis(arr_2.reshape(arr_2.shape[0], -1), maxk_b_ind, -1)
    # put maxk in arr_mask for plotting
    np.put_along_axis(arr_mask.reshape(arr_2.shape[0], -1), maxk_b_ind, maxk, -1)
    if reps > 1:
        maxk = maxk.repeat(reps, axis=0)
        maxk_b_ind = maxk_b_ind.repeat(reps, axis=0)

    # arr_3 = arr_3.reshape(arr_ref_fb.shape, -1)
    np.put_along_axis(arr_3.reshape(arr_3.shape[0], -1), maxk_b_ind, maxk, -1) # channel dimension
    return arr_mask.reshape(shape_arr_2), maxk, arr_3.reshape(shape_arr_3)


def replace_highest(arr_1, arr_2, arr_3, k=1):
    '''
    Replaces values from arr_2 to arr_3, according to highest k values
    in arr_1.
    Input:
        - arr_1: reference array to look up highest k values.
        - arr_2: source array, to extract values from.
        - arr_3: destination array, to inject values at given indices.
    Output:
        - arr_3 with vals from arr_2 at indices presenting arr_1 k-highest values. 
    '''
    # arr_1, arr_2, arr_3 = test_arrays()
    assert arr_1.size == arr_3.size, "Arrays mismatch"
    # 1. reshape with shape=(batch_size, H*W)
    arr_b = arr_1.reshape(arr_1.shape[0], -1)
    # 2. find indices for k highest values for each item along 1st dimension. 
    maxk_b_ind= np.argpartition(arr_b, -k)[:, -k:]

    # 3. flatten and unravel indices
    maxk_ind_flat = maxk_b_ind.flatten() #<- LET OP: unravelling flattened inds.
    maxk_ind_shape = np.unravel_index(maxk_ind_flat, arr_1.shape)
    # unravel: form indices coordinates system references to: (arr_1.shape)
    batch_indices = np.repeat(np.arange(arr_1.shape[0]), k) # (batch_size, k).
    maxk_indices = tuple([batch_indices] + [ind for ind in maxk_ind_shape])

    maxk = arr_2.reshape(arr_3.shape)[maxk_indices] # 3. resume this. 
    arr_3[maxk_indices] = maxk
    return maxk_indices, maxk, arr_3

def compute_delta(grand_zs, absolute=True):
    diff = np.zeros_like(grand_zs)

    for i in range(grand_zs.shape[0]):
        # i
        if absolute:
            diff[i] = np.abs(grand_zs[i] - np.mean(np.delete(grand_zs, i, axis=0), axis=0))
        else:
            diff[i] = grand_zs[i] - np.mean(np.delete(grand_zs, i, axis=0), axis=0)
    return diff

def craft_z(grand_zs, absolute, kept=None, reps=10, fold=False, device="cuda:0"):
    ''' Create z's from average, but with gaussian noise.
    Inputs:
        - nd_array: number-digits array with grand average of all z's spaces.
        - kept: integer or float. If integer, equals number of pixels to be kept
                                  if float, and fold=True, equals proportion of 
                                  pixels to be kept.
        - fold: dictates whether kept is read as absolute pixels count, or proportion.
    Outputs:
        - Artificial Z's for each digit. 
    '''
    # TODO: implement compute_delta(
    # mean = np.mean(grand_zs, axis=0) # should be ~= 0.
    # abs_diff = np.abs(grand_zs - mean)
    diff = compute_delta(grand_zs, absolute=absolute)

    batch_size = grand_zs.shape[0] * reps
    batch = torch.randn((batch_size, 1, 28, 28), dtype=torch.float32, device='cpu').numpy() # TODO: CHANGE 'CPU'
    arr_mask, _, batch = replace_highest_along_axis(diff, grand_zs, batch.copy(), kept)
    del diff
    return arr_mask, batch


def sample_from_crafted_z(net, all_nz, absolute, kept, reps, device, save_dir, monster_mode=False):
    ''' 
    Input:
        ndarray: n-dimensional but also number digit array.
    Output: plot.
    '''
    mask_zs, z = craft_z(all_nz, absolute=absolute, kept=kept, reps=reps)
    if monster_mode:
        B, C, H, W = z.shape
        hw = int(H * (B ** 0.5))
        z = z.reshape(1, C, hw, hw)
        # tile mask_zs horizontally (obtain 28 x 280 batches)
        mask_zs = np.tile(mask_zs, reps=(1, 1, 10))
        mask_zs.reshape(280, 280)
        kept = f'{kept}x{reps}'

        
    z = torch.from_numpy(z).to(device)
    x, _ = net(z, reverse=True)
    x = torch.sigmoid(x)
    
    images_concat = torchvision.utils.make_grid(x, nrow=int(x.shape[0] ** 0.5))
    torchvision.utils.save_image(images_concat, save_dir + f'/k{kept}_samples.png')
    plot_grand_z(mask_zs, save_dir + f'/k{kept}_mask.png')
    del x, z, images_concat, mask_zs
    # return x, z

def plot_grand_z(ndarray, filename, n_rows_cols=(2, 5)):
    # mpl.rc('text', usetex=True)
    # mpl.rcParams['text.latex.preamble']=[r"\boldmath"]

    n_rows, n_cols = n_rows_cols
    fig, axs = plt.subplots(n_rows, n_cols, sharex='all', sharey='all', figsize=(10, 7))

    n = 0
    for col in range(n_cols):
        for row in range(n_rows):
            axs[row, col].imshow(ndarray[n])
            ttl = r"Grand-${{z}}$ for {}".format(n)
            axs[row, col].title.set_text(ttl)
            n += 1
    
    fig.suptitle("Grand $bold{z}$ for each digit")
    fig.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    print('\nPlot saved to ' + filename)
    del ndarray

def y_distance_z(stats, n_categories=10):
    ''' take stats file, returns a similarity matrix
        for distances computed item-wise (pixel-wise).'''
    
    # z, y = label_zs(stats['z'])
    n_measures = 2
    out_size = [n_measures] + [stats['z'][0].shape[1], n_categories, n_categories]
    print(f'computing pixel-wise distances. Output matrix sized: {out_size}')
    pixelwise_mean = [np.mean(z, axis=0) for z in stats['z']]
    pixelwise_std = [np.std(z, axis=0) for z in stats['z']]

    y_distance_mean_std = np.zeros(out_size)
    
    for k, measure in enumerate([pixelwise_mean, pixelwise_std]):
        for d_i, m_i in enumerate(measure):
            for d_j, m_j in enumerate(measure):
                distance = np.linalg.norm(m_i - m_j)
                y_distance_mean_std[k, :, d_i, d_j] = distance
    
    return y_distance_mean_std


def calculate_distance(stats, joint=False, measure=None):
    distances = np.zeros(shape=(10, 10))
    if joint and measure:
        raise ValueError("set either joint or measure argument, not both.")

    if joint:
        joint_stats = []
        stds, means = stats['std'], stats['mean']
        for i, (std, mean) in enumerate(zip(stds, means)):
            std_mean = np.concatenate((std, mean), axis=1)

            for j, (std2, mean2) in enumerate(zip(stds, means)):
                std_mean2 = np.concatenate((std2, mean2), axis=1)

                min_rows = np.min((std_mean.shape[0], std_mean2.shape[0]))
                distance = np.linalg.norm(std_mean[:min_rows] - std_mean2[:min_rows])
                distances[i, j] = distance
    else:
        for i, m in enumerate(stats[measure]):
            for j, n in enumerate(stats[measure]):
                min_rows = np.min((m.shape[0], n.shape[0]))
                distance = np.linalg.norm(m[:min_rows] - n[:min_rows])
                distances[i, j] = distance
    return distances

def heatmap(square_mtx, filename, plot_title="Magnitude of distance between digits"):

    digits = [i for i in range(10)]

    fig, ax = plt.subplots()
    # configure main plot
    # show img, removing 0's to center color palette distribution. 
    norm_sq_mtx = square_mtx.copy()

    if square_mtx.shape[0] == 1: 
        # for new y_distance_z function
        # import ipdb; ipdb.set_trace()
        square_mtx = square_mtx.reshape(square_mtx.shape[1:])
        norm_sq_mtx = norm_sq_mtx.reshape(norm_sq_mtx.shape[1:])

    # remove diagonal
    diagonal_sel= np.array(digits)
    norm_sq_mtx[diagonal_sel, diagonal_sel] = None
    # subtract lowest value and divide
    norm_sq_mtx -= np.nanmin(norm_sq_mtx)
    norm_sq_mtx /= np.nanmax(norm_sq_mtx)
    im = ax.imshow(norm_sq_mtx, cmap="plasma")

    ax.set_xticks(digits)
    ax.set_yticks(digits)
    ax.set_xticklabels(digits)
    ax.set_yticklabels(digits)

    # annotate values within squares
    for i in range(10):
        for j in range(10):
            if i != j:
                val = norm_sq_mtx[i, j]
                col = 'w' if val < 0.6 else 'b'
                text = ax.text(j, i, "{:.2f}".format(square_mtx[i, j]),
                                   ha='center', va='center', color=col, size='x-small')
            else:
                break
    
    ax.set_title(plot_title)
    fig.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    print(f'Plot saved to {filename}')




def violin_eachdigit(stats, filename, m_epoch):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']
    digits = [i for i in range(10)]
    stds, means = [stats['std'], stats['mean']]

    fig = plt.figure()
    ax = fig.add_subplot()
    plt.title('z space stats for model at epoch {}'.format(m_epoch))
    plt.ylabel('Average $z$ values')
    plt.xlabel('Digits')
    
    # axs.set_title('Stats for epoch {} overview'.format(m_epoch))
    # 1. average std per digit
    std_each_digit = [np.mean(std) for std in stds]
    # 2. vector means for each digit
    mean_each_digit = [np.mean(m) for m in means]
    strings = ax.violinplot(dataset=means, positions=digits, showmedians=True, showextrema=False)

    for i, b in enumerate(strings['bodies']):
        b.set_facecolor(colors[i])
        b.set_edgecolor(color_variant(colors[i]))
        b.set_alpha(.6)

    '''
    axs[row,col].annotate('$\mu$: {:.2f}\n$\sigma^2$: {:.2f}'.format(ctrd_x, ctrd_y), (ctrd_x, ctrd_y))
    '''
    # fig.suptitle('Stats for epoch {} overview'.format(m_epoch))
    fig.tight_layout()
    # fig.subplots_adjust(top=.70) # .88
    #for dig in range(10):
    #	plt.scatter(means[dig], stds[dig], c=colors[dig], label=str(dig), alpha=.8, s=.8)
    plt.savefig(filename, bbox_inches='tight')
    print('\nPlot saved to ' + filename)



def scatter_eachdigit(stats, filename, m_epoch):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']
    stds, means = [stats['std'], stats['mean']]
    # Steps to replicate for each subplot:
    # plt.title('z space stats for model at epoch {}'.format(m_epoch))
    # plt.xlabel('mean')
    # plt.ylabel('std')
    nrows, ncols = (2, 5)
    fig, axs = plt.subplots(nrows, ncols, sharex='all', sharey='all', figsize=(10, 7))

    # axs.set_title('Stats for epoch {} overview'.format(m_epoch))
    dig = 0
    for col in range(ncols):
        for row in range(nrows):
            axs[row, col].scatter(means[dig], stds[dig], c=colors[dig],
                                      label=str(dig), alpha=.8, s=.8)
            # TODO: Plot average of all means and stds (per category).
            ctrd_x = np.mean(means[dig])
            ctrd_y = np.mean(stds[dig])
            axs[row, col].scatter(ctrd_x, ctrd_y, c=color_variant(colors[dig]))
            axs[row,col].set_xlabel('mean')
            axs[row,col].set_ylabel('std')
            axs[row,col].annotate('$\mu$: {:.2f}\n$\sigma^2$: {:.2f}'.format(ctrd_x, ctrd_y), (ctrd_x, ctrd_y))

            dig += 1

        if dig == 10: break

    fig.suptitle('Stats for epoch {} overview'.format(m_epoch))
    fig.tight_layout()
    fig.subplots_adjust(top=.88)
    #for dig in range(10):
    #	plt.scatter(means[dig], stds[dig], c=colors[dig], label=str(dig), alpha=.8, s=.8)
    plt.savefig(filename, bbox_inches='tight')
    print('\nPlot saved to ' + filename)


def color_variant(hex_color, brightness_offset=-50):
    """ takes a color like #87c95f and produces a lighter or darker variant """
    if len(hex_color) != 7:
        raise Exception("Passed %s into color_variant(), needs to be in #87c95f format." % hex_color)
    rgb_hex = [hex_color[x:x+2] for x in [1, 3, 5]]
    new_rgb_int = [int(hex_value, 16) + brightness_offset for hex_value in rgb_hex]
    new_rgb_int = [min([255, max([0, i])]) for i in new_rgb_int] # make sure new values are between 0 and 255
    # hex() produces "0x88", we want just "88"
    hex_string = [hex(i)[2:] for i in new_rgb_int]
    hex_string = [val if len(val) == 2 else "0" + val for val in hex_string]
    return "#" + "".join(hex_string)


def scatter_alldigits(stats, filename, m_epoch):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']
    fig, ax = plt.subplots()
    stds, means = [stats['std'], stats['mean']]
    plt.title('z space stats for model at epoch {}'.format(m_epoch))
    plt.xlabel('mean')
    plt.ylabel('std')

    for dig in range(10):
        ax.scatter(means[dig], stds[dig], c=colors[dig], label=str(dig), alpha=.8, s=.8)
    ax.legend()
    plt.savefig(filename, bbox_inches='tight')
    print('\nPlot saved to: ' + filename)



def grand_z(stats, filename=None, epoch_n=244):

    zspace = stats['z']
    

    for n in range(10):
        n_z = zspace[n]
        grand_nz = np.mean(n_z, axis=0)
        if 'all_nz' not in dir():
            all_nz = grand_nz
        else:
            all_nz = np.concatenate((all_nz, grand_nz))
    
    return all_nz
    


def track_distribution(net, loader, device, loss_fn, **kwargs):
    raise DeprecationWarning("replaced by track_z")
    net.eval()
    loss_meter = util.AverageMeter()
    bpd_meter = util.AverageMeter()
    
    # TODO - add:
    #with torch.nograd():
    with tqdm(total=len(loader.dataset)) as progress:

        digits_stds = [np.array([]).reshape(0, 1) for i in range(10)]
        digits_means = [np.array([]).reshape(0, 1) for i in range(10)]

        for x, y in loader:
            x = x.to(device)
            z, sldj = net(x, reverse=False)
            mean_z = z.mean(dim=1) # `1` is channel.
            std_z = z.std(dim=1)

            for digit in range(10):
                dig_std_z = std_z[ (y==digit).nonzero() ]
                dig_mean_z = mean_z[ (y==digit).nonzero() ]
                # here we concatenate each 'column array' to previous one (or initialized arr. of size=(0, 1)).
                digits_stds[digit] = np.concatenate(( digits_stds[digit], dig_std_z.to('cpu').detach().numpy() ))
                digits_means[digit] = np.concatenate(( digits_means[digit], dig_mean_z.to('cpu').detach().numpy() ))

            loss = loss_fn(z, sldj)
            loss_meter.update(loss.item(), x.size(0))
            bpd_meter.update(util.bits_per_dim(x, loss_meter.avg), x.size(0))

            progress.set_postfix(loss=loss_meter.avg, 
                                     bpd=bpd_meter.avg)
            progress.update(x.size(0))
    # add loss.avg() and bpd.avg() somewhere in the plot. 
    
    return {'std': digits_stds, 'mean': digits_means}


def track_z(net, loader, device, loss_fn, **kwargs):

    net.eval()
    loss_meter = util.AverageMeter()
    bpd_meter = util.AverageMeter()
    
    with tqdm(total=len(loader.dataset)) as progress:

        digits_stds = [np.array([]).reshape(0, 1) for i in range(10)]
        digits_means = [np.array([]).reshape(0, 1) for i in range(10)]

        ''' not quite but useful for reference. '''
        digits_z = [np.array([]).reshape(0, 1, 28, 28) for i in range(10)]

        for x, y in loader:
            x = x.to(device)
            z, sldj = net(x, reverse=False)
            mean_z = z.mean(dim=[1, 2, 3]) # `1` is channel.
            std_z = z.std(dim=[1, 2, 3])

            for n in range(10):
                # check if concatenation axis needs to be specified for ZZZZ.
                digit_indices = (y==n).nonzero()
                digit_indices_1dim = (y==n).nonzero(as_tuple=True)
                dig_std_z = std_z[ digit_indices ]
                dig_mean_z = mean_z[ digit_indices ]
                z_dig = z[digit_indices_1dim]
                # here we concatenate each 'array' to the previous one (or initialized arr. of size=(0, 1)).
                digits_stds[n] = np.concatenate(( digits_stds[n], dig_std_z.to('cpu').detach().numpy() ))
                digits_means[n] = np.concatenate(( digits_means[n], dig_mean_z.to('cpu').detach().numpy() ))
                # concatenate whole z space. 
                digits_z[n] = np.concatenate(( digits_z[n], z_dig.to('cpu').detach().numpy() ))

            loss = loss_fn(z, sldj)
            loss_meter.update(loss.item(), x.size(0))
            bpd_meter.update(util.bits_per_dim(x, loss_meter.avg), x.size(0))

            progress.set_postfix(loss=loss_meter.avg,
                                         bpd=bpd_meter.avg)
            progress.update(x.size(0))
    # add loss.avg() and bpd.avg() somewhere in the plot. 
    
    return {'std': digits_stds, 'mean': digits_means, 'z': digits_z}

def train(epoch, net, trainloader, device, optimizer, loss_fn, max_grad_norm):
    print('\nEpoch: %d' % epoch)
    net.train()
    loss_meter = util.AverageMeter()
    bpd_meter = util.AverageMeter()
    with tqdm(total=len(trainloader.dataset)) as progress_bar:
        for x, _ in trainloader:
            x = x.to(device)
            optimizer.zero_grad()
            z, sldj = net(x, reverse=False)
            loss = loss_fn(z, sldj)
            loss_meter.update(loss.item(), x.size(0))
            # import ipdb; ipdb.set_trace()
            loss.backward()
            util.clip_grad_norm(optimizer, max_grad_norm)
            optimizer.step()
            bpd_meter.update(util.bits_per_dim(x, loss_meter.avg))

            progress_bar.set_postfix(loss=loss_meter.avg,
                                                             bpd=bpd_meter.avg)
            progress_bar.update(x.size(0))
            #debugopt:
    return {'train_loss': loss_meter.avg, 
             'train_bpd': bpd_meter.avg}



def sample(net, batch_size, device):
    """Sample from RealNVP model.

    Args:
        net (torch.nn.DataParallel): The RealNVP model wrapped in DataParallel.
        batch_size (int): Number of samples to generate.
        device (torch.device): Device to use.
    """
    # side_size = 28 if net.in_channels == 1 else 32 # debatable.. but ok for now.
    z = torch.randn((batch_size, 1, 28, 28), dtype=torch.float32, device=device) #changed 3 -> 1
    x, _ = net(z, reverse=True)
    x = torch.sigmoid(x)

    return x, z


def test(epoch, net, testloader, device, loss_fn, num_samples, dir_samples, **args):
    global best_loss
    net.eval()
    loss_meter = util.AverageMeter()
    bpd_meter = util.AverageMeter()
    with torch.no_grad():
        with tqdm(total=len(testloader.dataset)) as progress_bar:
            for x, _ in testloader:
                x = x.to(device)
                z, sldj = net(x, reverse=False)
                ## HERE: TODO: do something with z and y
                loss = loss_fn(z, sldj)
                loss_meter.update(loss.item(), x.size(0))
                # bits per dimensions
                bpd_meter.update(util.bits_per_dim(x, loss_meter.avg), x.size(0))

                progress_bar.set_postfix(loss=loss_meter.avg,
                                                                 bpd=bpd_meter.avg)
                progress_bar.update(x.size(0))

    # Save checkpoint
    save_dir = dir_samples + f'/epoch_{epoch}'
    os.makedirs(save_dir, exist_ok=True)

    if loss_meter.avg < best_loss:
    # if epoch > 40:
        print('Saving...')
        state = {
            'net': net.state_dict(),
            'test_loss': loss_meter.avg,
            'epoch': epoch,
        }
        torch.save(state, save_dir + '/model.pth.tar')
        best_loss = loss_meter.avg

    # Save samples and data
    images, latent_z = sample(net, num_samples, device)

    # plot x and z
    images_concat = torchvision.utils.make_grid(images, nrow=int(num_samples ** 0.5), padding=2, pad_value=255)
    z_concat = torchvision.utils.make_grid(latent_z, nrow=int(num_samples ** 0.5), padding=2, pad_value=255)
    torchvision.utils.save_image(images_concat, save_dir+'/x.png')
    torchvision.utils.save_image(z_concat, save_dir+'/z.png')

    # with open(save_dir+'/z.pkl', 'wb') as z_serialize:
    # 	pickle.dump(latent_z, z_serialize)
    torch.save(latent_z, f = save_dir+'/z.pkl')


    # dict keys as returned by "train"
    train_loss = args['train_loss']
    train_bpd = args['train_bpd']
    report = [epoch, loss_meter.avg, bpd_meter.avg] + [train_loss, train_bpd]

    with open('{}/log'.format(dir_samples), 'a') as l:
        report = ", ".join([str(m) for m in report])
        report += "\n"
        print("\nWriting to disk:\n" + report + "At {}".format(dir_samples))
        l.write(report)

def verify_version(fp, version_string):
    ''' helper function to define what version of analysis was performed. 
    Used to select a model without given analysis version #.'''
    if os.path.isfile(fp):
        with open(fp, 'r+') as f:
            l = f.readline()

            while l:
                print("debugging value for l: " + l)

                if l.strip('\n') == version_string:
                    # if analysis # `version_string` was completed before:
                    print("Matched version: " + version_string)
                    return True
                    break
                else:
                    # TODO: fix eternal loop because f doesn't validate the while loop.
                    instead = l.strip()
                    print(f'Unmatched: {instead} != {version_string} ({fp}).', end='')
                    print(' Continuing...')
                l = f.readline()
        return False
    else: 
        print(f'File {fp} not found.')
        return False # if file doesn't exist.

def select_model(model_root, analyse_version, vmarker_fn='/version', 
                     epoch_range=(120, 440), test=False, granularity=10):
    ''' Select EPOCH according to version specifications. (change function name)
    Out: 
        - fp_vmarker : int --  file containing `version`
        - fp_model : str -- file to model.pth.tar
        '''
    verified_ver = True
    while verified_ver:
        if test:
            assert isinstance(test, int), 'Epoch must be int'
            model_epoch = test
        else:
            model_epoch = randrange(epoch_range[0], epoch_range[1], granularity) # TODO only produce n%10==0
        fp_model_root = model_root + '/epoch_' + str(model_epoch)
        fp_vmarker = fp_model_root + vmarker_fn
        if test:
            break
        else:
            verified_ver = verify_version(fp_vmarker, str(analyse_version))
    fp_model = fp_model_root + '/model.pth.tar'
    print('selected model at {}th epoch'.format(model_epoch))
    return fp_model_root, fp_model, fp_vmarker

def mark_version(version_str, fp_vmarker, finish=False, sep='-'):
    ''' write first and last parts of `version_str` to `fp_vmarker`
    Args:
        version_str
        fp_vmarker: str -- version marker file path
        finish: final call, else False
        sep: define first and last parts of `version_str`.
    '''
    vmarker = version_str.split(sep)
    m = open(fp_vmarker, 'a')
    if not finish:
        vmarker[0] += sep
        m.write('\n'+vmarker[0])
    else:
        m.write(vmarker[1]+'\n')
        # must end with a newline. byebye!
    m.close()

def load_network(model_dir, device, args):
    net = RealNVP( **filter_args(args.__dict__) )
    # assert os.path.isdir(model_dir), 'Error: no checkpoint directory found.'
    checkpoint = torch.load(model_dir)
    net = net.to(device)
    
    if str(device).startswith('cuda'):
        net = torch.nn.DataParallel(net, args.gpu_ids)
        cudnn.benchmark = args.benchmark
        try:
            net.load_state_dict(checkpoint['net'])
        except RuntimeError:
            raise ArchError('There is a problem importing the model, check parameters.')
    return net


class ArchError(Exception):
    def __init__(self, message=None):
        if not message:
            self.message = "State dictionary not matching your architecture. Check your params."
        else:
            self.message = message


def filter_args(arg_dict, desired_fields=None):
    """only pass to network architecture relevant fields."""
    if not desired_fields:
        desired_fields = ['net_type', 'num_scales', 'in_channels', 'mid_channels', 'num_levels']
    return {k:arg_dict[k] for k in desired_fields if k in arg_dict}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RealNVP on CIFAR-10')

    parser.add_argument('--benchmark', action='store_true', help='Turn on CUDNN benchmarking')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loader threads')
    parser.add_argument('--gpu_ids', default='[0]', type=eval, help='IDs of GPUs to use')
    parser.add_argument('--dataset', '-ds', default='MNIST', type=str, help='MNIST or CIFAR-10')
    # parser.add_argument('--num_samples', default=121, type=int, help='Number of samples at test time')

    # Analysis
    parser.add_argument('--force', '-f', action='store_true', default=False, help='Re-run z-space anal-yses.')
    parser.add_argument('--version', '-v', default='V-0.4', type=str, help='Analyses iteration')

    # General architecture parameters
    net_ = 'densenet'
    dataset_ = 'MNIST'
    # if dataset_.upper() == 'MNIST':
    in_channels_ = 1
    num_samples_ = 121
    if net_ == 'resnet':
        root_dir_ = 'data/res_3-8-32'
        batch_size_ = 256

        num_scales_ = 3
        mid_channels_ = 32
        num_levels_ = 8

    if net_ == 'densenet':
        # train one epoch: 6:14
        # test: 12:12
        num_scales_ = 3
        if dataset_ == 'celeba':
            batch_size_ = 64
            mid_channels_ = 128
            num_levels_ = 8
            num_samples_ = 64
            if gpus_ == '[0, 1]':
                batch_size_ = 16
                num_samples_ = 16

        elif dataset_.upper() == 'MNIST': # data/dense_test6
            root_dir_ = 'data/dense_test6'
            batch_size_ = 218
            mid_channels_ = 120
            num_levels_ = 10
            num_samples_ = 121
            num_scales_ = 3

    parser.add_argument('--net_type', default=net_, help='CNN architecture (resnet or densenet)')
    parser.add_argument('--batch_size', default=batch_size_, type=int, help='Batch size')
    parser.add_argument('--root_dir', default=root_dir_, help='Analyses root directory.')
    parser.add_argument('--num_scales', default=num_scales_, type=int, help='Real NVP multi-scale arch. recursions')
    parser.add_argument('--in_channels', default=in_channels_, type=int, help='dimensionality along Channels')
    parser.add_argument('--mid_channels', default=mid_channels_, type=int, help='N of feature maps for first resnet layer')
    parser.add_argument('--num_levels', default=num_levels_, type=int, help='N of residual blocks in resnet')


    main(parser.parse_args())
    # backends = ['GTK3Agg', 'GTK3Cairo', 'MacOSX', 'nbAgg', 'Qt4Agg', 'Qt4Cairo', 'Qt5Agg',
    # 		'Qt5Cairo', 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf',
    # 		'pgf', 'ps', 'svg', 'template']

# That's an impressive list of imports.
import numpy as np
import lutorpy as lua
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist

# We import sklearn.
import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

# We'll hack a bit with the t-SNE code in sklearn 0.15.2.
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import (_joint_probabilities,
                                    _kl_divergence)
from sklearn.utils.extmath import _ravel
# Random state.
RS = 20150101

# We'll use matplotlib for graphics.
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
# %matplotlib inline

# We import seaborn to make nice plots.
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})


def scatter(x, colors, unlabeled=None):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')

    if unlabeled is not None:
        sc = ax.scatter(unlabeled[:, 0], unlabeled[:, 1], lw=0, s=40,
                        c=[0.5, 0.5, 0.5])
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])

    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


# path = './results/new_trial_mnist_seeded'
path = './results/mnist_test'

epoch = 1


def loadFeats(filename):
    tbl = torch.load(filename)
    feats = tbl[0].asNumpyArray()
    label = tbl[1].asNumpyArray()
    for i in range(label.shape[0]):
        if label[i] == 10:
            label[i] = 0
    return feats, label

train_feats, train_labels = loadFeats(
    '%s/%sFeats_epoch%s' % (path, 'train', epoch))
test_feats, test_labels = loadFeats(
    '%s/%sFeats_epoch%s' % (path, 'test', epoch))

feats = np.concatenate([train_feats, test_feats], 0)
labels = np.concatenate([train_labels, test_labels], 0)
scatter(train_feats, train_labels, test_feats)
plt.savefig('digits_generated_%s_2d.png' % epoch, dpi=120)
scatter(feats, labels)
plt.savefig('digits_generated_%s_2d_all_labeled.png' % epoch, dpi=120)

# pca = sklearn.decomposition.PCA(2)
#
#
# pca.fit(feats)
# train_feats_pca = pca.transform(train_feats)
# test_feats_pca = pca.transform(test_feats)
#
# scatter(train_feats_pca, train_labels, test_feats_pca)
# plt.savefig('digits_generated_%s.png' % epoch, dpi=120)

# tsne = sklearn.manifold.TSNE(random_state=RS)
# feats = tsne.fit_transform(feats)
# scatter(feats, labels)
# plt.savefig('digits_generated_tsne.png', dpi=120)
# train_feats, test_feats = feats[
#     :train_labels.shape[0]], feats[train_labels.shape[0] + 1:]

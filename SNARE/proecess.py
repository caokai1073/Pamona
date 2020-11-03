import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import umap
import pandas as pd 


pamona_ra = np.loadtxt("pamona_fracs_rna_atac.txt")

pamona_ra_314 = np.loadtxt("pamona_fracs_rna_atac_314.txt")
pamona_ra_628 = np.loadtxt("pamona_fracs_rna_atac_628.txt")
pamona_ra_942 = np.loadtxt("pamona_fracs_rna_atac_924.txt")

scot_ra = np.loadtxt("scot_fracs_rna_atac.txt")

fig, ax = plt.subplots(figsize=(9,6))
plt.plot(np.arange(len(pamona_ra)), np.sort(pamona_ra), 'k', label='Pamona (0.157)')
plt.plot(np.arange(len(pamona_ra_314)), np.sort(pamona_ra_314), 'orange', label='Pamona: 30% correspondence (0.15)')
plt.plot(np.arange(len(pamona_ra_628)), np.sort(pamona_ra_628), 'red', label='Pamona: 60% correspondence (0.132)')
plt.plot(np.arange(len(pamona_ra_942)), np.sort(pamona_ra_942), 'purple', label='Pamona: 90% correspondence (0.116)')
plt.plot(np.arange(len(scot_ra)), np.sort(scot_ra), 'b' , label='SCOT (0.156)')
plt.tick_params(labelsize=25)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Arial') for label in labels]
plt.legend(prop={'size': 20, 'family' : 'Arial', })
fig.savefig('scatter0.eps',dpi=600,format='eps')

print(np.mean(pamona_ra))
print(np.mean(pamona_ra_314))
print(np.mean(pamona_ra_628))
print(np.mean(pamona_ra_942))
print(np.mean(scot_ra))





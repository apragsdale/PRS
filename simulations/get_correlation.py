import gzip, pandas
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import sys

ns = 400000
h2s = [0.67]
ms = [1000, 500, 200]
alpha = sys.argv[1]

correlations = {}
distributions_true = {}
distributions_infer = {}

def plot_distributions(vects, fname):
    (AFR_true, AFR_infer, EUR_true, EUR_infer, EAS_true, EAS_infer) = vects
#    true_mean = np.mean( np.concatenate(( AFR_true, EUR_true, EAS_true )) )
#    true_std = np.std( np.concatenate(( AFR_true, EUR_true, EAS_true )) )
#    infer_mean = np.mean( np.concatenate(( AFR_infer, EUR_infer, EAS_infer )) )
#    infer_std = np.std( np.concatenate(( AFR_infer, EUR_infer, EAS_infer )) )
#    AFR_true = (AFR_true - true_mean) / true_std
#    EUR_true = (EUR_true - true_mean) / true_std
#    EAS_true = (EAS_true - true_mean) / true_std
#    AFR_infer = (AFR_infer - infer_mean) / infer_std
#    EUR_infer = (EUR_true - infer_mean) / infer_std
#    EAS_infer = (EAS_true - infer_mean) / infer_std

#    AFR_true = np.random.choice(AFR_true, 10000, replace=False)
#    EUR_true = np.random.choice(EUR_true, 10000, replace=False)
#    EAS_true = np.random.choice(EAS_true, 10000, replace=False)
#    AFR_infer = np.random.choice(AFR_infer, 10000, replace=False)
#    EUR_infer = np.random.choice(EUR_infer, 10000, replace=False)
#    EAS_infer = np.random.choice(EAS_infer, 10000, replace=False)

    fig = plt.figure(7591, figsize=(7.5,5))
    fig.clf()
    ax1 = plt.subplot2grid((2,3),(0,0))
    ax2 = plt.subplot2grid((2,3),(1,0))
    sns.distplot(AFR_true, color='red', hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 1}, 
                 label = 'AFR', ax=ax1)
    sns.distplot(EAS_true, color='purple', hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 1},
                 label = 'EAS', ax=ax1)
    sns.distplot(EUR_true, color='blue', hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 1},
                 label = 'EUR', ax=ax1)
    sns.distplot(AFR_infer, color='red', hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 1},
                 label = 'AFR', ax=ax2)
    sns.distplot(EAS_infer, color='purple', hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 1},
                 label = 'EAS', ax=ax2)
    sns.distplot(EUR_infer, color='blue', hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 1},
                 label = 'EUR', ax=ax2)
    ax1.set_xlabel('True PRS')
    ax1.legend(frameon=False)
    ax2.set_xlabel('Inferred PRS')
    ax2.legend(frameon=False)
    
    ax3 = plt.subplot2grid((2,3),(0,1),colspan=2, rowspan=2)
    ax3.plot(EUR_true, EUR_infer, 'o', markersize=1, c='blue', label='EUR', rasterized=True)
    ax3.plot(EAS_true, EAS_infer, 'o', markersize=1, c='purple', label='EAS', rasterized=True)
    ax3.plot(AFR_true, AFR_infer, 'o', markersize=1, c='red', label='AFR', rasterized=True)
    ax3.set_xlabel('True PRS')
    ax3.set_ylabel('Inferred PRS')
    ax3.legend(loc=1)
    
    fig.tight_layout()
    fig.savefig(fname)



for h2 in h2s:
    correlations[h2] = {}
    distributions_true[h2] = {}
    distributions_infer[h2] = {}
    for m in ms:
        correlations[h2][m] = {'AFR':[], 'EAS':[], 'EUR':[]}
        distributions_true[h2][m] = {'AFR':{'mean':[],'var':[]}, 'EAS':{'mean':[],'var':[]}, 'EUR':{'mean':[],'var':[]}}
        distributions_infer[h2][m] = {'AFR':{'mean':[],'var':[]}, 'EAS':{'mean':[],'var':[]}, 'EUR':{'mean':[],'var':[]}}
        for i in range(1,101):
            fname = f'data/sim{i}_nhaps_{ns}_{ns}_{ns}_h2_{h2}_m_{m}_alpha_{alpha}.prs.gz'
:q           try:
                data = pandas.read_csv(fname, sep='\t')
                AFR_true = data[data['Pop']=='AFR']['PRS_true']
                AFR_infer = data[data['Pop']=='AFR']['PRS_infer']
                EUR_data = data[data['Pop']=='EUR']  
                EUR_data = EUR_data[EUR_data.isnull().any(axis=1)]
                EUR_true = EUR_data[EUR_data['Pop']=='EUR']['PRS_true']
                EUR_infer = EUR_data[EUR_data['Pop']=='EUR']['PRS_infer']
                EAS_true = data[data['Pop']=='EAS']['PRS_true']
                EAS_infer = data[data['Pop']=='EAS']['PRS_infer']
                correlations[h2][m]['AFR'].append(np.corrcoef(AFR_true, AFR_infer)[0][1])
                correlations[h2][m]['EUR'].append(np.corrcoef(EUR_true, EUR_infer)[0][1])
                correlations[h2][m]['EAS'].append(np.corrcoef(EAS_true, EAS_infer)[0][1])
                distributions_true[h2][m]['AFR']['mean'].append(np.mean(AFR_true))
                distributions_true[h2][m]['AFR']['var'].append(np.var(AFR_true))
                distributions_true[h2][m]['EUR']['mean'].append(np.mean(EUR_true))
                distributions_true[h2][m]['EUR']['var'].append(np.var(EUR_true))
                distributions_true[h2][m]['EAS']['mean'].append(np.mean(EAS_true))
                distributions_true[h2][m]['EAS']['var'].append(np.var(EAS_true))
                distributions_infer[h2][m]['AFR']['mean'].append(np.mean(AFR_infer))
                distributions_infer[h2][m]['AFR']['var'].append(np.var(AFR_infer))
                distributions_infer[h2][m]['EUR']['mean'].append(np.mean(EUR_infer))
                distributions_infer[h2][m]['EUR']['var'].append(np.var(EUR_infer))
                distributions_infer[h2][m]['EAS']['mean'].append(np.mean(EAS_infer))
                distributions_infer[h2][m]['EAS']['var'].append(np.var(EAS_infer))
                if h2 == 0.67 and m == 1000:
                    plot_distributions([AFR_true, AFR_infer, EUR_true, EUR_infer, EAS_true, EAS_infer], f'plots/distributions_h2_{h2}_m_{m}_sim{i}.pdf')
            except IOError:
                print("no file ", alpha, i, h2, m)
                continue

"""

def z_score(mu_focal, var_focal, mu_target):
    return (mu_target-mu_focal) / np.sqrt(var_focal)

Zscores = {}
Zscores['EAS'] = {}
Zscores['AFR'] = {}

for h2 in h2s:
    Zscores['EAS'][h2] = {}
    Zscores['AFR'][h2] = {}
    for m in ms:
        Zscores['EAS'][h2][m] = []
        Zscores['AFR'][h2][m] = []
        for i in range(len(distributions_true[h2][m]['AFR']['mean'])):
            Zscores['AFR'][h2][m].append(
                (z_score(distributions_true[h2][m]['EUR']['mean'][i],
                         distributions_true[h2][m]['EUR']['var'][i],
                         distributions_true[h2][m]['AFR']['mean'][i]),
                    z_score(distributions_infer[h2][m]['EUR']['mean'][i],
                            distributions_infer[h2][m]['EUR']['var'][i],
                            distributions_infer[h2][m]['AFR']['mean'][i])))
            Zscores['EAS'][h2][m].append(
                (z_score(distributions_true[h2][m]['EUR']['mean'][i],
                         distributions_true[h2][m]['EUR']['var'][i],
                         distributions_true[h2][m]['EAS']['mean'][i]),
                    z_score(distributions_infer[h2][m]['EUR']['mean'][i],
                            distributions_infer[h2][m]['EUR']['var'][i],
                            distributions_infer[h2][m]['EAS']['mean'][i])))

# plot Z scores
h2_ms_pop = [(0.67,200,'EAS'),(0.67,500,'EAS'),(0.67,1000,'EAS'),
    (0.67,200,'AFR'),(0.67,500,'AFR'),(0.67,1000,'AFR')]

fig = plt.figure(54210, figsize=(10,6))
fig.clf()

for i,vals in enumerate(h2_ms_pop):
    ax = plt.subplot(2,3,i+1)
    h2,m,pop = vals
    xs = [x[0] for x in Zscores[pop][h2][m]]
    ys = [x[1] for x in Zscores[pop][h2][m]]
    r = np.corrcoef(xs,ys)[0][1]
    ax.scatter(xs, ys)
    ax.set_title('{0}, h2={1}, ncausal={2}'.format(pop, h2, m), fontsize=8)
    ax.set_xlabel(r'$Z_{true}$', fontsize=8)
    ax.set_ylabel(r'$Z_{PRS}$', fontsize=8)
    xlims, ylims = ax.get_xlim(), ax.get_ylim()
    ax.text(xlims[0] + .2*(xlims[1]-xlims[0]),
            ylims[0] + .8*(ylims[1]-ylims[0]),
            f'r={r:.2}', ha='center', va='center', fontsize=8)

fig.tight_layout()
fig.savefig(f'zscore_h2_0.67_alpha_{alpha}.pdf')


# plot Z scores
h2_ms_pop = [(0.33,200,'EAS'),(0.33,500,'EAS'),(0.33,1000,'EAS'),
    (0.33,200,'AFR'),(0.33,500,'AFR'),(0.33,1000,'AFR')]

fig = plt.figure(53481, figsize=(10,6))
fig.clf()

for i,vals in enumerate(h2_ms_pop):
    ax = plt.subplot(2,3,i+1)
    h2,m,pop = vals
    xs = [x[0] for x in Zscores[pop][h2][m]]
    ys = [x[1] for x in Zscores[pop][h2][m]]
    if len(xs) > 0:
        r = np.corrcoef(xs,ys)[0][1]
        ax.scatter(xs, ys)
        ax.set_title('{0}, h2={1}, ncausal={2}'.format(pop, h2, m), fontsize=8)
        ax.set_xlabel(r'$Z_{true}$', fontsize=8)
        ax.set_ylabel(r'$Z_{PRS}$', fontsize=8)
        xlims, ylims = ax.get_xlim(), ax.get_ylim()
        ax.text(xlims[0] + .2*(xlims[1]-xlims[0]),
                ylims[0] + .8*(ylims[1]-ylims[0]),
                f'r={r:.2}', ha='center', va='center', fontsize=8)

fig.tight_layout()
fig.savefig(f'zscore_h2_0.33_alpha_{alpha}.pdf')



# from https://matplotlib.org/3.1.1/gallery/statistics/customized_violin.html

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')



fig = plt.figure(1, figsize=(10,6))
fig.clf()

for j,h2 in enumerate(h2s):
    for i,m in enumerate(ms):
        ax1 = plt.subplot(2,3,3*j+i+1)
        
        data = [correlations[h2][m][k] for k in ['AFR','EAS','EUR']]
        if len(data[0]) == 0:
            continue
        
        for y in [0.2,0.4,0.6,0.8]:
            ax1.axhline(y, ls='--', lw=1, color='black')

        parts = ax1.violinplot(data,showmeans=False, showmedians=False,
                               showextrema=False)
        
        for pc in parts['bodies']:
            pc.set_facecolor('lightblue')
            pc.set_edgecolor('black')
            pc.set_alpha(1)

        quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)
        whiskers = np.array([
            adjacent_values(sorted_array, q1, q3)
            for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
        whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]

        inds = np.arange(1, len(medians) + 1)
        ax1.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
        ax1.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
        ax1.vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)

        # set style for the axes
        labels = ['AFR','EAS','EUR']
        for ax in [ax1]:
            set_axis_style(ax, labels)

        ax1.set_ylim([0,1])
        ax1.set_title(r'$h^2 = {1}, m={0}$'.format(m, h2))
        ax1.set_ylabel("Pearson's correlation")
        ax1.set_xlabel("Population")

plt.tight_layout()

plt.savefig(f'violins_alpha_{alpha}.pdf')
#plt.show()
"""

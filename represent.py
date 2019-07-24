import sys 
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

colors = sns.color_palette("bright", 10)

plt.style.use('ggplot')
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['legend.frameon'] = 'True'

def AIC_BIC_plot(components_range, AIC, BIC, figs_dir):
  plt.clf()
  plt.errorbar(components_range, BIC, marker='o', color='blue', label='BIC', alpha=0.5)
  plt.errorbar(components_range, AIC, marker='x', color='red', label='AIC', alpha=0.5)
  plt.xlabel('num of components')
  plt.ylabel('Measure')
  plt.xticks(components_range)
  plt.legend(loc='best')
  plt.savefig('{}/AIC_BIC_components.png'.format(figs_dir), bbox_inches='tight')

font_size = 8
def plot_paradoxical_feature(plotX, plotY, fig_dir, ind_var, Y_var, groups, gr_coefficient, gr_confidence, coefficient, confidence=0):
    plt.clf()

    number_of_components = len(gr_coefficient)
    coefs_togather = [coefficient]
    coefs_togather.extend(gr_coefficient)
    confi_togather = [confidence]
    confi_togather.extend(gr_confidence)
    colors_togather = ['gray']
    labels_togather = ['MLR']
    for ind in range(number_of_components):
    	labels_togather.append('# ' + str(ind + 1))
    	colors_togather.append(colors[ind])


    plt.bar(range(number_of_components + 1), coefs_togather, 0.35, yerr=confi_togather, color=colors_togather)
    plt.xlabel('Coefficient for {}'.format(ind_var))
    plt.xticks(range(number_of_components + 1))
    plt.gca().set_xticklabels(labels_togather)
    plt.savefig('{}/variable_{}.png'.format(fig_dir, ind_var))


def show_components_normalized(mu, name_of_variables, Y_var, Y_values, y_confidence, Y_pred, y_sigma, var_dict, fig_dir, bbox=None):  
    number_of_components, number_of_features = mu.shape
    zav = (2 * np.pi) / number_of_features

    mapped_var_index = dict()
    for ind in range(number_of_features):
        mapped_var_index[ind] = np.max(mu[:, ind]) - np.min(mu[:, ind])

    list_of_index = sorted(mapped_var_index, key=lambda k: mapped_var_index[k], reverse=True)
    labels = []
    for ind in range(number_of_features):
        labels.append(var_dict[name_of_variables[list_of_index[ind]]])

    plt.clf()

    max_val = [-1 * sys.float_info.max for ind in range(number_of_features)]
    min_val = [sys.float_info.max for ind in range(number_of_features)]
    for com_num in range(number_of_components):
        for ind in range(number_of_features):
            point = mu[com_num][list_of_index[ind]]
            if point > max_val[list_of_index[ind]]:
                max_val[list_of_index[ind]] = point
            if point < min_val[list_of_index[ind]]:
                min_val[list_of_index[ind]] = point

    for com_num in range(number_of_components):
        x = []
        y = []
        for ind in range(number_of_features):
            lng = float(mu[com_num][list_of_index[ind]]) / float(max_val[list_of_index[ind]])
            

            x.append(np.cos(ind * zav) * lng)
            y.append(np.sin(ind * zav) * lng)
        x.append(x[0])
        y.append(y[0])

        plt.fill(x, y, color=colors[com_num], alpha=0.2, label='{0}={1:.2f} ({2:.3f})'.format(Y_var, Y_values[com_num], y_confidence[com_num]))

    axis_points_x = []
    axis_points_y = []
    for ind in range(number_of_features):
        x = np.cos(ind * zav)
        y = np.sin(ind * zav)
        plt.plot([0, x], [0, y], color='grey', linestyle=':', alpha=0.5)

        if ind * zav < (np.pi / 2):
            plt.text(x, y, labels[ind], {'ha': 'left', 'va': 'bottom', 'fontsize': 12}, rotation= ind * float(360.0 / number_of_features))
        elif ind * zav < np.pi:
            plt.text(x, y, labels[ind], {'ha': 'right', 'va': 'bottom', 'fontsize': 12}, rotation= ind * float(360.0 / number_of_features))
        elif ind * zav < 1.5 * np.pi:
            plt.text(x, y, labels[ind], {'ha': 'right', 'va': 'top', 'fontsize': 12}, rotation= ind * float(360.0 / number_of_features))
        else:
            plt.text(x, y, labels[ind], {'ha': 'left', 'va': 'top', 'fontsize': 12}, rotation= ind * float(360.0 / number_of_features))

    plt.box(False)
    plt.axis('off')
    if bbox:
        plt.legend(loc='best', bbox_to_anchor=bbox)
    else:
        plt.legend(loc='best')
    # plt.legend(loc='best', bbox_to_anchor=(0.8, 0.6))
    plt.savefig('{}'.format(fig_dir), bbox_inches='tight')

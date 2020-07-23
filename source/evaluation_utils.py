import numpy as np
from eval_proposal import ANETproposal
import matplotlib.pylab as plt


def run_evaluation(ground_truth_file, json_results_file,
                   max_avg_n_proposals=100,
                   tiou_thresholds=np.linspace(0.5, 0.95, 10),
                   subset='validation'):

    anet_proposal = ANETproposal(ground_truth_file, json_results_file,
                                 tiou_thresholds=tiou_thresholds,
                                 max_avg_nr_proposals=max_avg_n_proposals,
                                 subset=subset, verbose=True,
                                 check_status=False)
    anet_proposal.evaluate()
    recall = anet_proposal.recall
    average_recall = anet_proposal.avg_recall
    average_n_proposals = anet_proposal.proposals_per_video

    return average_n_proposals, average_recall, recall


def plot_metric(average_n_proposals, average_recall, recall,
                tiou_thresholds=np.linspace(0.5, 0.95, 10),
                fig_file='../figures/figure_results.png'):

    fn_size = 14
    plt.figure(num=None, figsize=(6, 5))
    ax = plt.subplot(1, 1, 1)
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33',
              '#a65628', '#f781bf', '#999999', '#737373']

    area_under_curve = np.zeros_like(tiou_thresholds)
    for i in range(recall.shape[0]):
        area_under_curve[i] = np.trapz(recall[i], average_n_proposals)

    for idx, tiou in enumerate(tiou_thresholds[::2]):
        ax.plot(average_n_proposals, recall[2 * idx, :], color=colors[idx + 1],
                label=f'tiou=[{tiou:.1f}], area='
                      f'{int(area_under_curve[2 * idx] * 100) / 100.}',
                linewidth=4, linestyle='--', marker=None)

    a = int(np.trapz(average_recall, average_n_proposals) * 100) / 100.
    ax.plot(average_n_proposals, average_recall, color=colors[0],
            label=f'tiou=0.5:0.05:0.95, area={a}',
            linewidth=4, linestyle='-', marker=None)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[-1]] + handles[:-1], [labels[-1]] + labels[:-1],
              loc='best')

    plt.ylabel('Average Recall', fontsize=fn_size)
    plt.xlabel('Average Number of Proposals per Video', fontsize=fn_size)
    plt.grid(b=True, which="both")
    plt.ylim([0, 1.0])
    plt.setp(plt.axes().get_xticklabels(), fontsize=fn_size)
    plt.setp(plt.axes().get_yticklabels(), fontsize=fn_size)
    plt.savefig(fig_file)

    return
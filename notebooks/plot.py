import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def read_log(filename, verbose=False):
    if verbose:
        print('Filename:', filename)
    with open(filename, 'r') as f:
        lines = f.readlines()
    iterations = []
    optimizer = []
    penalties = []
    objectives = []
    fs = []
    
    for i in range(len(lines)):
        
        if 'Iteration' in lines[i]:
            iterations.append(int(lines[i].split()[2]))
            optimizer.append(lines[i+1].split()[1])
            penalties.append(float(lines[i+2].split()[2]))
            objectives.append(float(lines[i+3].split()[2]))
            fs.append(float(lines[i+4].split()[2]))
        
            if verbose:
                print('Iteration {:6d}: Penalty: {:.6f}, Objective: {:.6f}, F = {:.8f}'.format(iterations[-1], penalties[-1], objectives[-1], fs[-1]))
    return iterations, optimizer, penalties, objectives, fs

def plot_logs(x_list, y1_list, y2_list, labels, title, x_label, y1_label, y2_label, y1_colors=None, y2_colors=None, legend_filename=None, plot_filename=None):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel(x_label)
   
    if y1_colors is None:
        y1_colors = ['blue'] * len(y1_list)
    if y2_colors is None:
        y2_colors = ['red'] * len(y2_list)

    for i, (x, y1, color) in enumerate(zip(x_list, y1_list, y1_colors)):
        ax1.plot(x, y1, color=color, linewidth=2, label=f'{labels[i]} - {y1_label}')
    ax1.set_ylabel(y1_label, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(alpha=0.2, axis='y', color='gray')

    ax2 = ax1.twinx()
    for i, (x, y2, color) in enumerate(zip(x_list, y2_list, y2_colors)):
        ax2.plot(x, y2, color=color, linewidth=2, label=f'{labels[i]} - {y2_label}', linestyle='dashed')
    ax2.set_ylabel(y2_label, color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.grid(alpha=0.2, axis='y', color='gray')
    
    ax1.axvline(x=230, color='black', linewidth=2, linestyle='--') 
   
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
    
    
    
    plt.title(title)
    plt.tight_layout()
    
    if plot_filename:
        fig.savefig(plot_filename, bbox_inches='tight', dpi=300)
        
    plt.show()
    
    all_lines = lines1 + lines2
    all_labels = labels1 + labels2
    if legend_filename:
        fig_legend = plt.figure(figsize=(10, 1.5))
        fig_legend.legend(all_lines, all_labels, loc='center', ncol=4, fontsize=10)
        fig_legend.savefig(legend_filename, bbox_inches='tight')
        plt.close(fig_legend)
        

# Cargar datos del primer archivo
# Prostate

filenames= [
    '../Results/Log/Prostate/01.log',
    '../Results/Log/Prostate/02.log',
    '../Results/Log/Prostate/04.log',
    '../Results/Log/Prostate/05.log',
]


labels = [
    'Prostate_01',
    'Prostate_02',
    'Prostate_03',
    'Prostate_04'

]

# # Head and Neck
# filenames= [
#     '../Results/Log/Head-and-Neck/01.log',
#     '../Results/Log/Head-and-Neck/02.log',
#     '../Results/Log/Head-and-Neck/03.log',
#     '../Results/Log/Head-and-Neck/09.log',
#     '../Results/Log/Head-and-Neck/11.log',
#     '../Results/Log/Head-and-Neck/13.log',
#     '../Results/Log/Head-and-Neck/14.log',
#     '../Results/Log/Head-and-Neck/15.log'
# ]


# labels = [
#     'HaN_01',
#     'HaN_02',
#     'HaN_03',
#     'HaN_04',
#     'HaN_05',
#     'HaN_06',
#     'HaN_07',
#     'HaN_08'
# ]

#Liver

# filenames= [
#     '../Code_POO/multicore/l01.log',
#     '../Code_POO/multicore/l02.log',
#    '../Code_POO/multicore/l10.log',
   
# ]


# labels = [
#     'Liver_01',
#     'Liver_02',
#     'Liver_03',
# ]


start = 0
end = 8

x_list = []
y1_list = []  # F
y2_list = []  # Objective
y3_list = []  # Penalty
y4_list = []  # Optimizer

for f in filenames:
    print(f'Processing file: {f}')
    iterations, optimizer, penalties, objectives, fs = read_log(f)

    x_list.append(iterations[start:end])
    y1_list.append(fs[start:end])
    y2_list.append(objectives[start:end])
    y3_list.append(penalties[start:end])
    y4_list.append(optimizer[start:end])



num_patients = len(filenames)

cmap1 = cm.get_cmap('tab10', num_patients)
cmap2 = cm.get_cmap('Set1', num_patients)

y1_colors = [cmap1(i) for i in range(num_patients)]  # Para F
y2_colors = [cmap2(i) for i in range(num_patients)]  # Para Objective o Penalty


plot_logs(
    x_list=x_list,
    y1_list=y1_list,
    y2_list=y2_list,
    labels=labels,
    title="Comparison of Curves",
    x_label="Iterations",
    y1_label="F",
    y2_label="Objective",
    y1_colors=y1_colors,
    y2_colors=y2_colors,
    legend_filename="legend_F_Objective.png",
    plot_filename="plot_F_Objective.png"
)

plot_logs(
    x_list=x_list,
    y1_list=y1_list,
    y2_list=y3_list,
    labels=labels,
    title="Comparison of Curves",
    x_label="Iterations",
    y1_label="F",
    y2_label="Penalty",
    y1_colors=y1_colors,
    y2_colors=y2_colors,
    legend_filename="legend_F_Objective.png",
    plot_filename="plot_F.png"
)
# iterations1, penalties1, objectives1, fs1 = read_log(filenames[0])
# start = 0
# end = 120
# iterations1 = iterations1[start:end]
# penalties1 = penalties1[start:end]
# objectives1 = objectives1[start:end]
# fs1 = fs1[start:end]

# # Cargar datos del segundo archivo
# iterations2, penalties2, objectives2, fs2 = read_log(filenames[1])
# iterations2 = iterations2[start:end]
# penalties2 = penalties2[start:end]
# objectives2 = objectives2[start:end]
# fs2 = fs2[start:end]

# labels = ['HaN_01', 'HaN_02']  

# # Graficar
# plot_logs(
#     x_list=[iterations1, iterations2],
#     y1_list=[objectives1, objectives2],
#     y2_list=[penalties1, penalties2],
#     labels=labels,
#     title="Comparison of Curves",
#     x_label="Iterations",
#     y1_label="Objetive",
#     y2_label="Penalty",
#     y1_colors=['blue', 'green'], 
#     y2_colors=['orange', 'red'] 
# )

# plot_logs(
#     x_list=[iterations1, iterations2],
#     y1_list=[fs1, fs2],
#     y2_list=[objectives1, objectives2],
#     labels=labels,
#     title="Comparison of Curves",
#     x_label="Iterations",
#     y1_label="F",
#     y2_label="Objetive",
#     y1_colors=['blue', 'green'], 
#     y2_colors=['orange', 'red'] 
# )

# plot_logs(
#     x_list=[iterations1, iterations2],
#     y1_list=[fs1, fs2],
#     y2_list=[penalties1, penalties2],
#     labels=labels,
#     title="Comparison of Curves",
#     x_label="Iterations",
#     y1_label="F",
#     y2_label="Penalty",
#     y1_colors=['blue', 'green'], 
#     y2_colors=['orange', 'red'] 
# )
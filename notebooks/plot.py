import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def read_log(filename, verbose=False):
    if verbose:
        print('Filename:', filename)
    with open(filename, 'r') as f:
        lines = f.readlines()
    iterations = []
    penalties = []
    objectives = []
    fs = []
    
    for i in range(len(lines)):
        if 'Iteration' in lines[i]:
            iterations.append(int(lines[i].split()[2]))
            penalties.append(float(lines[i+1].split()[2]))
            objectives.append(float(lines[i+2].split()[2]))
            fs.append(float(lines[i+3].split()[2]))
            if verbose:
                print('Iteration {:6d}: Penalty: {:.6f}, Objective: {:.6f}, F = {:.8f}'.format(iterations[-1], penalties[-1], objectives[-1], fs[-1]))
    return iterations, penalties, objectives, fs

def plot_logs(x_list, y1_list, y2_list, labels, title, x_label, y1_label, y2_label, y1_colors=None, y2_colors=None):
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
        ax2.plot(x, y2, color=color, linewidth=2, label=f'{labels[i]} - {y2_label}')
    ax2.set_ylabel(y2_label, color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.grid(alpha=0.2, axis='y', color='gray')
    
   
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
    
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Cargar datos del primer archivo

filenames= [
    '../Code_POO/multicore/p_01.log',
    '../Code_POO/multicore/p_02.log',
    '../Code_POO/multicore/p_03.log',
]

labels = [
    'HaN_01',
    'HaN_02',
    'HaN_03'
]

start = 0
end = 120

x_list = []
y1_list = []  # F
y2_list = []  # Objective
y3_list = []  # Penalty

for f in filenames:
    print(f'Processing file: {f}')
    iterations, penalties, objectives, fs = read_log(f)
    
    x_list.append(iterations[start:end])
    y1_list.append(fs[start:end])
    y2_list.append(objectives[start:end])
    y3_list.append(penalties[start:end])

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
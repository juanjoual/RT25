import matplotlib.pyplot as plt


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
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', bbox_to_anchor=(1, 0.05), fontsize=10)
    
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Cargar datos del primer archivo
filename1 = '../NIO_POO/Adam_log/3_08.log'
iterations1, penalties1, objectives1, fs1 = read_log(filename1)
start = 0
end = 120
iterations1 = iterations1[start:end]
penalties1 = penalties1[start:end]
objectives1 = objectives1[start:end]
fs1 = fs1[start:end]

# Cargar datos del primer archivo
filename2 = '../NIO_POO/Adam_log/3_09.log'
iterations2, penalties2, objectives2, fs2 = read_log(filename2)
start = 0
end = 120
iterations2 = iterations2[start:end]
penalties2 = penalties2[start:end]
objectives2 = objectives2[start:end]
fs2 = fs2[start:end]



# Cargar datos del segundo archivo
filename3 = '../NIO_POO/SGD_log/3.log'  
iterations3, penalties3, objectives3, fs3 = read_log(filename3)
iterations3 = iterations3[start:end]
penalties3 = penalties3[start:end]
objectives3 = objectives3[start:end]
fs3 = fs3[start:end]

labels = ['Adam_0.99', 'Adam_0.9', 'SGD']  

# Graficar
plot_logs(
    x_list=[iterations1, iterations2, iterations3],
    y1_list=[objectives1, objectives2, objectives3],
    y2_list=[penalties1, penalties2, penalties3],
    labels=labels,
    title="Comparison of Curves",
    x_label="Iterations",
    y1_label="Objetive",
    y2_label="Penalty",
    y1_colors=['blue', 'green', 'black'], 
    y2_colors=['orange', 'red', 'black'] 
)

plot_logs(
    x_list=[iterations1, iterations2, iterations3],
    y1_list=[fs1, fs2, fs3],
    y2_list=[objectives1, objectives2, objectives3],
    labels=labels,
    title="Comparison of Curves",
    x_label="Iterations",
    y1_label="F",
    y2_label="Objetive",
    y1_colors=['blue', 'green', 'black'], 
    y2_colors=['orange', 'red', 'black'] 
)

plot_logs(
    x_list=[iterations1, iterations2, iterations3],
    y1_list=[fs1, fs2, fs3],
    y2_list=[penalties1, penalties2, penalties3],
    labels=labels,
    title="Comparison of Curves",
    x_label="Iterations",
    y1_label="F",
    y2_label="Penalty",
    y1_colors=['blue', 'green', 'black'], 
    y2_colors=['orange', 'red', 'black'] 
)
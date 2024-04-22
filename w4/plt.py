import matplotlib.pyplot as plt

# Training data without pruning or quantization
epochs_no_pruning_quant = list(range(1, 31))
losses_no_pruning_quant = [
    6.18572, 3.22144, 2.15521, 1.47079, 1.00015,
    0.73344, 0.59117, 0.17186, 0.04416, 0.02579,
    0.01804, 0.01369, 0.01173, 0.00957, 0.00683,
    0.00605, 0.00558, 0.00529, 0.00518, 0.00483,
    0.00481, 0.00405, 0.00420, 0.00423, 0.00405,
    0.00402, 0.00384, 0.00401, 0.00376, 0.00399
]

# Creating the plot
plt.figure(figsize=(10, 6))
plt.plot(epochs_no_pruning_quant, losses_no_pruning_quant, marker='o', linestyle='-', color='blue', label='original -loss')
plt.title('training loss - original')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid(True)
plt.legend()
plt.show()

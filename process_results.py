results_file = open("tanh_loss_results.txt", "r")

results = []
for line in results_file:
    if (len(line.split(" ")) > 1):
        results.append(line.split(" ")[-1].strip())
        print(line.split(" ")[-1].strip())

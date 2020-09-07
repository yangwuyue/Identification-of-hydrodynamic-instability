import matplotlib.pyplot as plt


content = open("RB_result.txt").readlines()

labels = set()
values = []

for line in content:
    value = line.strip().split("),(")
    for item in value:
        labels.add(float(item.split(",")[1][2:]))
    values.append(value)

labels = sorted(list(labels))

color = ['k*-']
label = ['RB']

index = 0
for value in values:
    x = []
    y = []
    turn_point = None
    for item in value:
        v_item = item.split(",")
        x.append(labels.index(float(v_item[1][2:])))
        y.append(float(v_item[0]))
        if len(y) > 1 and (y[-1]-0.5)*(y[-2]-0.5) < 0:
            turn_point = labels[x[-1]]
    print(label[index], "==> ", turn_point)
    plt.plot(x, y, color[index],label=label[index])
    plt.legend()
    index += 1
show_labels = []
x = range(len(labels))
for index in range(len(labels)):
    if index % 8 == 0:
        show_labels.append(str(round(labels[index],1)))
    else:
        show_labels.append("")
plt.xticks(x, show_labels, rotation=90)
plt.title("RB prediction")
plt.xlabel("Ra")
plt.ylabel("Prediction")
plt.show()

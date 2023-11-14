import matplotlib.pyplot as plt

def get_num(s):
    return int(s.split(":")[1])

c0_y = []
c1_y = []
x_axis = list(range(2123))

with open("freq.txt", "r") as f:
    for line in f:
        x = line.split(",")
        c0_y.append(get_num(x[0]))
        c1_y.append(get_num(x[1]))

plt.plot(x_axis, c0_y)
plt.show()

# plt.plot(x_axis, c1_y)
# plt.show()
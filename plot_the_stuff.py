import matplotlib.pyplot as plt

lines = open("best_values.txt", "r").read().split(',')[:-1]
values = list(map(int, lines))
xs = [0.1 * x for x in range(1, len(values) + 1)]

plt.plot(xs, values)
plt.ylim([1596771532, 1598449253])
# plt.plot(xs, [1597463007] * len(values))
plt.show()

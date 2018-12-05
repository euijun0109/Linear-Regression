import LinearRegression as lr

train = [(1, 3), (2, 6), (3, 7), (6, 9), (4, 8), (5, 6)]
LR = lr.LinearRegression(0.1)
LR.run(10000, train)


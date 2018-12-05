import matplotlib.pyplot as plt
class LinearRegression:
    def __init__(self, m, alpha):
        self.theta1 = 0
        self.theta2 = 0
        self.m = m
        self.cost1 = 0
        self.cost2 = 0
        self.alpha = alpha
        self.mseSum = 0

    def hypothesis(self, x):
        return self.theta1 + self.theta2 * x

    def batch(self, pair):
        for x, y in pair:
            self.cost1 += self.hypothesis(x) - y
            self.cost2 += (self.hypothesis(x) - y) * x
            #calculates mse also
            self.mseSum += (y - self.hypothesis(x)) ** 2

    def GD(self):
        newTheta1 = self.theta1 - self.alpha * (1/self.m) * (self.cost1)
        newTheta2 = self.theta2 - self.alpha * (1/self.m) * (self.cost2)
        self.theta1 = newTheta1
        self.theta2 = newTheta2

    def calculate(self, *train):
        for pair in train:
            self.batch(pair)
        self.GD()
        print("costs:", self.cost1, ", ", self.cost2)
        print("mse: ", self.mseSum / self.m)
        self.mseSum = 0
        self.cost1 = 0
        self.cost2 = 0
        print("theta 1: ", self.theta1, "theta 2: ", self.theta2, '\n')

    def run(self, i, train):
        for _ in range(i):
            self.calculate(train)
        self.display(train)

    def display(self, data):
        x = []
        y = []
        for pair in data:
            x.append(pair[0])
            y.append(pair[1])
        X = [0, x[self.m - 1] + 1]
        Y = [self.theta1 + self.theta2 * 0, self.theta1 + self.theta2 * (x[self.m - 1] + 1)]
        plt.plot(x, y, "ro", label="data")
        plt.plot(X, Y, label="best fit line")
        plt.legend()
        plt.show()
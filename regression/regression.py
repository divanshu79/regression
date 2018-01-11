from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

# xs = np.array([1,2,3,4,5,6,7,8,9], dtype=np.float64)
# ys = np.array([1,3,2,6,3,6,4,7,8], dtype=np.float64)

def create_db(hm, variance, step, correlation=False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(1,variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]
    return np.array(xs,dtype=np.float64), np.array(ys,dtype=np.float64)

def bestfitslop_intercept(xs,ys):
    m = (((mean(xs)*mean(ys))-(mean(xs*ys)))/((mean(xs)**2)-(mean(xs**2))))
    b = mean(ys)-m*mean(xs)
    return m, b

def squared_error(ys_orig, ys_line):
    return sum((ys_line-ys_orig)**2)

def cof_of_determination(ys_orig,ys_line):
    # y_mean_line = [mean(ys_orig) for y in ys_orig]
    y_mean_line = mean(ys_orig)
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1-(squared_error_regr/squared_error_y_mean)

xs,ys = create_db(10,400,10,correlation='pos')
# print(xs)
# print(ys)
m, b = bestfitslop_intercept(xs, ys)

# print(m,b)

regression_line = [(m*x)+b for x in xs]

pridict_x = 15
pridict_y = (m*pridict_x)+b

r_squared = cof_of_determination(ys, regression_line)
print(r_squared)

plt.scatter(xs, ys)
plt.scatter(pridict_x, pridict_y)
plt.plot(xs,regression_line)
plt.show()
print(pridict_y)

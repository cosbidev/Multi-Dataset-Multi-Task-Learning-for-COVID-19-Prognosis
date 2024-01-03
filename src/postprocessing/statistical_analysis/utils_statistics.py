#define custom function
def g_mean(x):
    a = np.log(x)
    return np.exp(a.mean())

#calculate geometric mean
g_mean([value1, value2, value3, ...])
import matplotlib.pylab as plt
def plot(kernel, name=''):
    plt.imshow(kernel)
    plt.colorbar()
    plt.title(name)
    plt.show()
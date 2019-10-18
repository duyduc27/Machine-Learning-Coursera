import matplotlib.pyplot as plt
def plotData(x, y):
    """Plots the data points x and y into a new figure    
    PLOTDATA(x,y) plots the data points and gives the figure axes labels of
    population and profit."""

    plt.scatter(x, y, marker='x', c='r')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.show()
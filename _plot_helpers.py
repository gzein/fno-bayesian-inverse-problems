import matplotlib.pyplot as plt


def plot_XY(x, y, X_name='X', Y_name='Y'):
    plt.figure(figsize=(12, 4))

    plt.scatter(x, y)
    plt.xlabel(X_name)
    plt.ylabel(Y_name)
    plt.title(f"{X_name} vs {Y_name} Plot")
    plt.legend()
    plt.grid(True)
    plt.show(block=False)
    input("Press Enter to continue...")  # Keep the plot open until user presses Enter
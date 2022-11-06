import matplotlib.pyplot as plt
import numpy as np
import torch
import torchquad as tq




def main():
    f = lambda x: torch.sin(x)
    a = 0
    b = 1
    n = 10
    print(trapezoid(f, a, b, n))
def f(x):
    return torch.sin(x)

def trapezoid(f, a, b, n):
    """Trapezoid rule for numerical integration.
    Args:
        f (callable): Function to integrate.
        a (float): Lower bound of integration.
        b (float): Upper bound of integration.
        n (int): Number of intervals.
    Returns:
        float: Integral approximation.
    """
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return np.trapz(y, x)




def plot_convergence(evals, fvals, ground_truth, labels, dpi=150):
    """Plots errors vs. function evaluations (fevals) and shows the convergence rate.

    Args:
        evals (list of np.array): Number of evaluations, for each method a np.array of ints.
        fvals (list of np.array): Function values for evals.
        ground_truth (np.array): Ground truth values.
        labels (list): Method names.
        dpi (int, optional): Plot dpi. Defaults to 150.
    """
    plt.figure(dpi=dpi)
    for evals_item, f_item, label in zip(evals, fvals, labels):
        evals_item = np.array(evals_item)
        abs_err = np.abs(np.asarray(f_item) - np.asarray(ground_truth))
        abs_err_delta = np.mean(np.abs((abs_err[:-1]) / (abs_err[1:] + 1e-16)))
        label = label + "\nConvergence Rate: " + str.format("{:.2e}", abs_err_delta)
        plt.semilogy(evals_item, abs_err, label=label)

    plt.legend(fontsize=6)
    plt.xlabel("# of function evaluations")
    plt.ylabel("Absolute error")



if __name__ == "__main__":
    main()
#Recived help from ChatGPT
#refernced Exam1 and numericalmethod file by Dr. Smay
#worked with Brian Nguyen in a group
"""
Simulates the gravel production process.

Steps:
1. Define probability density functions (PDFs) for log-normal and truncated log-normal distributions.
2. Compute cumulative distribution function (CDF) limits using numerical integration.
3. Generate random samples based on the truncated log-normal distribution.
4. Compute statistical properties (mean and variance) of the samples.
5. Display the results.
"""

import math
import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


def ln_PDF(D, mu, sig):
    """
    Computes the probability density function (PDF) of a log-normal distribution.

    Steps:
    1. Check if D is zero to prevent division errors.
    2. Compute the PDF using the log-normal formula.
    3. Return the computed probability density value.

    :param D: Rock diameter.
    :param mu: Mean of ln(D).
    :param sig: Standard deviation of ln(D).
    :return: Probability density f(D).
    """
    if D == 0.0:
        return 0.0
    p = 1 / (D * sig * math.sqrt(2 * math.pi))
    _exp = -((math.log(D) - mu) ** 2) / (2 * sig ** 2)
    return p * math.exp(_exp)


def tln_PDF(D, mu, sig, F_DMin, F_DMax):
    """
    Computes the truncated log-normal PDF.

    Steps:
    1. Compute the standard log-normal PDF.
    2. Normalize it using the cumulative distribution limits.
    3. Return the truncated PDF value.

    :param D: Rock diameter.
    :param mu: Mean of ln(D).
    :param sig: Standard deviation of ln(D).
    :param F_DMin: CDF at DMin.
    :param F_DMax: CDF at DMax.
    :return: Truncated log-normal PDF.
    """
    return ln_PDF(D, mu, sig) / (F_DMax - F_DMin)


def F_tlnpdf(D, mu, sig, D_Min, D_Max, F_DMax, F_DMin):
    """
    Integrates the truncated log-normal PDF from D_Min to D to find probability.

    Steps:
    1. Check if D is within valid range.
    2. Compute the integral of the truncated log-normal PDF.
    3. Return the computed probability value.

    :param D: Upper bound for integration.
    :return: Probability value.
    """
    if D > D_Max or D < D_Min:
        return 0
    result, _ = quad(lambda x: tln_PDF(x, mu, sig, F_DMin, F_DMax), D_Min, D)
    return result


def makeSample(ln_Mean, ln_sig, D_Min, D_Max, F_DMax, F_DMin, N=100):
    """
    Generates a sample of rock sizes based on the truncated log-normal PDF.

    Steps:
    1. Generate N random probabilities.
    2. Solve for the corresponding rock diameters using fsolve.
    3. Return the list of rock sizes.

    :param N: Number of samples.
    :return: List of rock sizes.
    """
    probs = np.random.uniform(0, 1, N)
    d_s = [fsolve(lambda D: F_tlnpdf(D, ln_Mean, ln_sig, D_Min, D_Max, F_DMax, F_DMin) - p, D_Min)[0] for p in probs]
    return d_s


def sampleStats(D):
    """
    Computes the mean and variance of a given sample.

    Steps:
    1. Compute the mean of the dataset.
    2. Compute the unbiased variance.
    3. Return both values.

    :param D: List of values.
    :return: Tuple (mean, variance).
    """
    mean = np.mean(D)
    var = np.var(D, ddof=1)  # Unbiased variance
    return mean, var


def getFDMaxFDMin(mean_ln, sig_ln, D_Min, D_Max):
    """
    Computes F_DMax and F_DMin using quad integration over the log-normal distribution.

    Steps:
    1. Compute the cumulative distribution function (CDF) at D_Max.
    2. Compute the CDF at D_Min.
    3. Return both values.

    :return: (F_DMin, F_DMax)
    """
    F_DMax, _ = quad(lambda D: ln_PDF(D, mean_ln, sig_ln), 0, D_Max)
    F_DMin, _ = quad(lambda D: ln_PDF(D, mean_ln, sig_ln), 0, D_Min)
    return F_DMin, F_DMax


def main():
    """
    Simulates the gravel production process with user inputs.

    Steps:
    1. Define initial parameters.
    2. Compute cumulative distribution limits.
    3. Generate samples and compute their statistics.
    4. Display computed values.
    """
    mean_ln = math.log(2)
    sig_ln = 1
    D_Max = 1
    D_Min = 3.0 / 8.0
    N_samples = 11
    N_sampleSize = 100

    F_DMin, F_DMax = getFDMaxFDMin(mean_ln, sig_ln, D_Min, D_Max)

    Samples, Means = [], []
    for _ in range(N_samples):
        sample = makeSample(mean_ln, sig_ln, D_Min, D_Max, F_DMax, F_DMin, N=N_sampleSize)
        Samples.append(sample)
        mean, var = sampleStats(sample)
        Means.append(mean)
        print(f"Sample: mean = {mean:.3f}, variance = {var:.3f}")

    mean_of_means, var_of_means = sampleStats(Means)
    print(f"Mean of sample means: {mean_of_means:.3f}")
    print(f"Variance of sample means: {var_of_means:.6f}")


if __name__ == '__main__':
    main()

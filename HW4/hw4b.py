#Recived help from ChatGPT and Deepseek AI
#refernced hw4a.py file by Dr. Smay
#worked with Gin Muang in a group, as well as collaborated with Sean Ross albeit not in a group capacity

# region imports
import numpy as np  # For numerical calculations
import matplotlib
matplotlib.use('TkAgg')  # Switch to TkAgg backend for plotting (fixes some display issues)
import matplotlib.pyplot as plt  # For creating plots
from scipy.stats import lognorm  # To use log-normal distribution functions
from scipy.integrate import cumulative_trapezoid  # For calculating the cumulative distribution (CDF)
# endregion

# region functions
def plot_truncated_lognormal(mu, sigma, d_min, d_max):
    """
    Plot the truncated log-normal distribution's PDF and CDF.

    This function take in the parrameter of the log-normal distribution (mu, sigma) and the truncation range (d_min, d_max). then we calculate the upper limit for integration, which is set at 75% of the total range. Then the the  truncated PDF and CDF are calculated over the range from d_min to this upper limit. The function will plot the PDF with a shaded area up to the 75th percentile and label it. The CDF is also plotted, with vertical and horizontal line marking where the 75% occurs.
    """
    # Compute the upper integration limit for the truncated distribution
    d_trunc = d_min + (d_max - d_min) * 0.75

    # Create a normalized x-axis (from 0 to 1) and map it to the actual D values over the truncated range
    x_norm = np.linspace(0, 1, 500)  # Normalized x-axis
    D = d_min + x_norm * (d_trunc - d_min)  # Map normalized x values to actual D values

    # Calculate the full log-normal PDF over the range D
    pdf_full = lognorm.pdf(D, s=sigma, scale=np.exp(mu))

    # Normalize the PDF over the truncated range [d_min, d_trunc]
    cdf_dmin = lognorm.cdf(d_min, s=sigma, scale=np.exp(mu))
    cdf_dtrunc = lognorm.cdf(d_trunc, s=sigma, scale=np.exp(mu))
    norm_factor = cdf_dtrunc - cdf_dmin  # Normalization factor to ensure the total probability sums to 1
    truncated_pdf = pdf_full / norm_factor  # Truncated PDF

    # Use numerical integration to calculate the truncated CDF
    cdf_numeric = cumulative_trapezoid(truncated_pdf, D, initial=0)

    # Find D* where the CDF reaches 0.75, and get its corresponding normalized x value
    D_star = np.interp(0.75, cdf_numeric, D)
    x_star = (D_star - d_min) / (d_trunc - d_min)

    # Create two subplots that share the same x-axis for better comparison
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    # --- Top Plot: Truncated PDF ---
    axes[0].plot(x_norm, truncated_pdf, label="Truncated PDF", color='blue', lw=2)
    axes[0].fill_between(x_norm, truncated_pdf, where=(x_norm <= x_star), color='grey', alpha=0.5)
    axes[0].set_ylabel("f(D)")  # Label for the y-axis
    axes[0].set_title("Truncated Log-Normal PDF (Normalized x)")  # Title for the plot

    # Add an annotation to explain the PDF equation and probability at D*
    eq_str = r'$f(D)=\frac{1}{D\,\sigma\sqrt{2\pi}}\exp\left[-\frac{1}{2}\left(\frac{\ln(D)-\mu}{\sigma}\right)^2\right]$'
    prob_str = r'$P(D<{:.2f}\;|\;TLN({:.2f},{:.2f},{:.3f},{:.3f}))=0.75$'.format(D_star, mu, sigma, d_min, d_max)
    annotation_text = eq_str + "\n" + prob_str  # Combine the equation and probability for display

    x_text = 0.5 * x_star  # Position of the text in the plot
    y_text = np.interp(x_text, x_norm, truncated_pdf)
    arrow_target = (x_star, 0.5 * np.interp(x_star, x_norm, truncated_pdf))
    axes[0].annotate(annotation_text, xy=arrow_target, xytext=(x_text, y_text),
                     arrowprops=dict(arrowstyle="->", color='black'), fontsize=10, horizontalalignment='left')

    # --- Bottom Plot: Truncated CDF ---
    axes[1].plot(x_norm, cdf_numeric, label="Truncated CDF", color='blue', lw=2)
    axes[1].axvline(x=x_star, color='black', linestyle='-', linewidth=1)  # Vertical line at D*
    axes[1].axhline(y=0.75, color='black', linestyle='-', linewidth=1)  # Horizontal line at 75%
    axes[1].plot(x_star, 0.75, 'o', markerfacecolor='white', markeredgecolor='red')  # Mark the 75% point
    axes[1].set_xlabel("Normalized x")  # x-axis label for CDF
    axes[1].set_ylabel(r'$\theta(x)=\int_{D_{\min}}^{D} f(D)\,dD$')  # y-axis label for CDF
    axes[1].set_title("Truncated Log-Normal CDF")  # Title for the CDF plot

    # Add grid and legend to both subplots for better readability
    for ax in axes:
        ax.grid(True)
        ax.legend()

    plt.tight_layout()  # Adjust layout for neat spacing between plots
    plt.show()  # Display the plots

# endregion

# region function calls
if __name__ == "__main__":
    # Ask the user for inputs (parameters of the log-normal distribution and truncation range)
    mu = float(input("Enter mu for log-normal distribution: "))  # e.g., 0.69
    sigma = float(input("Enter sigma for log-normal distribution: "))  # e.g., 1.00
    d_min = float(input("Enter D_min for truncation: "))  # e.g., 0.047
    d_max = float(input("Enter D_max for pre-sieved distribution: "))  # e.g., 0.244

    # Generate and display the plots
    plot_truncated_lognormal(mu, sigma, d_min, d_max)
# endregion
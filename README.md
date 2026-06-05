# Functional Data Analysis (FDA) Package

The goal of this project is to develop a robust Python library for performing a wide-range of Functional Data Analysis (FDA) techniques at scale and with options for user customization. Existing package is limited but contains most of the standard building blocks for performing a simple FDA. The long-term vision of this package is to eventually develop capabilities for tailoring FDA techniques to specific industries and their needs (e.g. healthcare, finance, etc.) in order to expand understanding of how FDA can be used in practice and hopefully, through the journey, shed a little more love to an often understudied corner of data science!

## Features
- **Basis Functions**: Represent functional objects using B-Spline or Fourier bases.
- **Smoothing**: Convert discrete points to functional data curves using least squares and penalized smoothing.
- **Descriptive Statistics**: Compute functional mean, functional variance, and functional covariance.
- **Functional PCA**: Extract the main modes of variation via FPCA.

## Installation
Run:
```bash
pip install -q -e ..
```

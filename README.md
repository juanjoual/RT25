# Radiotherapy 2025

## 1. TROTS

### 1.1. Links

- [Homepage](https://sebastiaanbreedveld.nl/trots/)

- [Detailed description](https://sebastiaanbreedveld.nl/trots/TROTS_Data_Description.pdf)

- [Example code (mostly MATLAB)](https://github.com/SebastiaanBreedveld/TROTS)

### 1.2. Matrix types (data.matrix.Type):

- 0: Normal (fluence-to-dose domain).
- 1: Fluence domain only (smoothing).
- 2: From the manual: *"This is a quadratic or square matrix, meaning that when
the problem is extended with auxiliary decision variables (e.g. to solve
mini-max problems when minimising a pointwise maximum), the padding
should be done both in rows and columns."*

For now we will only export Type 0 matrices.

### 
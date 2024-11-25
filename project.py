import numpy as np
import matplotlib.pyplot as plt
import math

# Constants
cp = 1005  # Specific heat at constant pressure (J/kg.K)
R = 287  # Specific gas constant for air (J/kg.K)
T_inlet = 25 + 273.15  # Inlet temperature (K)
p_inlet = 101325  # Inlet pressure (Pa)
rho_inlet = p_inlet / (R * T_inlet)  # Inlet density (kg/m^3)
Cx_inlet = 136  # Axial velocity at inlet (m/s)
r_hub = 0.9 / 2  # Hub radius (m)
r_tip = 1 / 2  # Blade tip radius (m)
m_dot = rho_inlet * Cx_inlet * np.pi * (r_tip**2 - r_hub**2)  # Mass flow rate (kg/s)
alpha_inlet = 30 * (np.pi/180)  # Inlet flow angle (rad)
N = 6000 * (2*np.pi/60) # Rotational speed (rad/s)

deltarC_hub = 82.3  # Hub swirl increase (m^2/s)
deltarC_tip = 84.4  # Tip swirl increase (m^2/s)

# Grid generation
r_width = 0.1  # Blade width (m)

res = 10  # Grid resolution (nodes/blade)

N = int(res / r_width)
M = int(res / r_width)
x = np.linspace(0, 1, N)
r = np.linspace(r_hub, r_tip, M)
X, R = np.meshgrid(x, r)

# Initialize stream function
Psi = (R**2 - r_hub**2) / (r_tip**2 - r_hub**2)

# Initialize density, inlet mass flow, and inlet total enthalpy
rho = np.full_like(Psi, rho_inlet)
H0_inlet = cp * T_inlet + Cx_inlet**2 / 2

# Calculate velocities at each node
Cx = np.full_like(Psi, Cx_inlet)
Cr = np.zeros_like(Psi)

dPsi_dr = np.gradient(Psi, axis=0) / (2 * (x[1] - x[0]))
dPsi_dx = np.gradient(Psi, axis=1) / (2 * (r[1] - r[0]))

Cx = (m_dot/(2*np.pi*rho*r))*dPsi_dr
Cr = (-m_dot/(2*np.pi*rho*r))*dPsi_dx


#================================================================================================
# Initialize rC at each node based on the inlet conditions
rC = np.zeros_like(Psi)

# Loop over the radial positions to set rC as a constant along each radial line
for j in range(M):
    rC[j, :] = r[j] * Cx_inlet * np.tan(alpha_inlet)

#================================================================================================
# LE and TE for Blade
LE = math.floor((len(x)-res)/2)  # Index for the leading edge
TE = math.floor((len(x)+res)/2)  # Index for the trailing edge

# Initialize loss factor matrix with zeros
varpi = np.zeros_like(Psi)

# Set uniform loss factor from LE+1 to TE
for i in range(LE + 1, TE + 1):
    varpi[ : , i] = 0.5 / (TE - LE)

#================================================================================================

# Initialize additional swirl increase array at the trailing edge
delta_rC_TE = np.zeros(M)

# Calculate delta_rC_TE for each radial station from hub to shroud
for j in range(M):
    delta_rC_TE[j] = deltarC_hub + (deltarC_tip - deltarC_hub) * (j / (M - 1))

# Distribute the additional swirl inside the blade from LE+1 to TE
for j in range(M):
    for i in range(LE + 1, TE + 1):
        rC[j,i] = rC[j, i - 1] + delta_rC_TE[j] / (TE - LE)

# Update rC after the trailing edge
for i in range(TE+1, N):
    rC[:,i] = rC[:, TE]


#================================================================================================

# Initialize vorticity and auxiliary terms

omega = np.zeros_like(Psi)
S = np.zeros_like(Psi)  
T = np.zeros_like(Psi)
for j in range(M):
    C_theta = rC[j, :] / r[j]
    V_squared = Cx**2 + Cr**2 + C_theta**2
    H0 = 0
    T = (H0_inlet / cp) - (V_squared / (2 * cp))

# Calculate vorticity ω for each node based on the formula provided
for i in range(1, N):  # Avoid boundary points
    for j in range(1, M-1):  # Avoid boundary points
        rC_theta_diff = (rC[j+1, i] - rC[j-1,i])
        S_diff = S[ j+1, i] - S[ j-1, i]
        H_diff = (H0_inlet / rho_inlet)
        

        omega[j, i] = (np.pi / ((r[j]-r[j-1]) * m_dot * Cx[j,i])) * (( C_theta/ (r[j]) * rC_theta_diff) + T[j,i] * S_diff - H_diff)
        
print(T_inlet)
#================================================================================================

# Plot the updated vorticity distribution
plt.figure()
plt.contourf(X, R, omega, levels=500, cmap='viridis')
plt.colorbar(label='Vorticity (ω)')
plt.xlabel('x')
plt.ylabel('r')
plt.title('Vorticity Distribution (ω)')

plt.show()
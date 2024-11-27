import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
blade_width = 0.1  # m
r_hub = 0.9
r_tip = 1.0
num_stations = 10
rho_inlet = 1.5  # kg/m^3
T_inlet = 25 + 273.15  # K
R_gas = 287  # J/(kg*K)
P0_inlet = 1e5  # Pa
Cp = 1005  # J/(kg*K)
Cx_inlet = 136  # m/s
alpha_inlet = np.deg2rad(30)  # radians

delta_rC_theta_TE_hub = 82.3  # m^2/s
delta_rC_theta_TE_tip = 84.4  # m^2/s

iter_max = 5
N = 3 * num_stations
M = num_stations
LE = num_stations
TE = LE + num_stations

# Grid Generation
def generate_grid():
    x = np.linspace(-1.5, 1.5, N)
    r = np.linspace(r_hub, r_tip, M)
    return np.meshgrid(x, r)

X, R = generate_grid()

# Initialization
def initialize_variables():
    Psi = (R**2 - r_hub**2) / (r_tip**2 - r_hub**2)
    rho = np.full_like(Psi, rho_inlet)
    p = np.full_like(Psi, P0_inlet)
    T = np.full_like(Psi, T_inlet)
    H0_inlet = Cp * T_inlet + 0.5 * Cx_inlet**2
    H0 = np.full_like(Psi, H0_inlet)
    m_dot = rho_inlet * Cx_inlet * np.pi * (r_tip**2 - r_hub**2)
    S = np.zeros_like(Psi)
    return Psi, rho, p, T, H0_inlet, H0, m_dot, S


# Swirl Distribution
def start_rotor(delta_rC_theta_TE_hub, delta_rC_theta_TE_tip, Psi):
    # Initialize rC
    rC = np.zeros_like(Psi)

    for j in range(M):
        rC[j, :] = R[j] * Cx_inlet * np.tan(alpha_inlet)
    
    # Initialize additional swirl increase array at TE
    delta_rC_TE = np.zeros(M)

    # Calculate delta_rC_TE for each radial station from hub to shroud
    for j in range(M):
        delta_rC_TE[j] = delta_rC_theta_TE_hub + (delta_rC_theta_TE_tip - delta_rC_theta_TE_hub) * (j  / (M - 1))
    
    # Distribute the additional swirl inside the blade from LE+1 to TE
    for j in range(M):
        for i in range(LE + 1, TE+1):
            rC[j, i] = rC[j, i - 1] + delta_rC_TE[j] * (i - LE) / (TE - LE)   
    
    # Update rC after the trailing edge
    for i in range(TE+1, N):
        rC[:,i] = rC[:, TE]

    return rC



def calculate_vorticity(Psi, rC_theta, R, T, S, H0):
    omega = np.zeros_like(Psi)
    for i in range(1, N):
        for j in range(1, M-1):
            # Calculate vorticity using finite differences and interpolated values
            term1 = rC_theta[j + 1, i] - rC_theta[j - 1, i]
            term2 = S[j + 1, i] - S[j - 1, i] 
            term3 = H0[j + 1, i] - H0[j - 1, i] 
            
            omega[j, i] = (np.pi / (Cx_inlet * (R[j,i]-R[j-1,i]))) * (((C_theta[j, i] / R[j, i]) *  term1) + T[j,i] * term2 - term3)
    return omega

def update_stream_function(Psi, rho, R, omega, dx):

    new_Psi = Psi.copy()
    
    for j in range(1, N - 1):  # Loop over axial positions
        for i in range(1, M - 1):  # Loop over radial positions

            # Interpolate rho and R at half-grid points
            # 1p1h = 1 plus 1 half
            # 1m1h = 1 minus 1 half
            rho_ip1h_j = (rho[i + 1, j] + rho[i, j]) / 2
            rho_im1h_j = (rho[i - 1, j] + rho[i, j]) / 2
            rho_i_jp1h = (rho[i, j + 1] + rho[i, j]) / 2
            rho_i_jm1h = (rho[i, j - 1] + rho[i, j]) / 2

            R_ip1h_j = (R[i + 1, j] + R[i, j]) / 2
            R_im1h_j = (R[i - 1, j] + R[i, j]) / 2
            R_i_jp1h = (R[i, j + 1] + R[i, j]) / 2
            R_i_jm1h = (R[i, j - 1] + R[i, j]) / 2
            
            # Calculate A_ij using interpolated values
            A_ij = 1 / (
                1 / (rho_ip1h_j * R_ip1h_j)
                + 1 / (rho_im1h_j * R_im1h_j)
                + 1 / (rho_i_jp1h * R_i_jp1h)
                + 1 / (rho_i_jm1h * R_i_jm1h)
            )
            
            # Calculate B_ij using interpolated values
            B_ij = (
                Psi[i + 1, j] / (rho_ip1h_j * R_ip1h_j)
                + Psi[i - 1, j] / (rho_im1h_j * R_im1h_j)
                + Psi[i, j + 1] / (rho_i_jp1h * R_i_jp1h)
                + Psi[i, j - 1] / (rho_i_jm1h * R_i_jm1h)
            )
            
            # Update Psi
            new_Psi[i, j] = A_ij * (B_ij + dx**2 * omega[i, j])
            
    return new_Psi

def calculate_velocities(Psi, m_dot, R, rho):
    Cx = np.zeros_like(Psi)
    Cr = np.zeros_like(Psi)
    Cm = np.zeros_like(Psi)

    dx = 0.01
    dr = 0.01
    
    for i in range(1, N-1):
        for j in range(1, M-1):
            # Calculate axial and radial velocities at internal points
            Cx[j, i] = m_dot / (2 * np.pi * R[j, i] * rho[j, i]) * (Psi[j + 1, i] - Psi[j - 1, i]) / (2 * dr)
            Cr[j, i] = -m_dot / (2 * np.pi * R[j, i] * rho[j, i]) * (Psi[j, i + 1] - Psi[j, i - 1]) / (2 * dx)
            Cm[j, i] = np.sqrt(Cx[j, i]**2 + Cr[j, i]**2)
            
        # Calculate Cx at hub and shroud
        Cx[0, i] = m_dot / (2 * np.pi * R[0, i] * rho[0, i]) * (Psi[1, i] - Psi[0, i])/ (2 * dr)
        Cx[M - 1, i] = -m_dot / (2 * np.pi * R[M - 1, i] * rho[M - 1, i]) * (Psi[M - 1, i] - Psi[M - 2, i]) / (2 *dr)
    
    # Set Cr = 0 at the walls
    Cr[0, :] = 0
    Cr[M - 1, :] = 0

    Cx[0, N-1:N] = Cx[0, N-2]

    return Cx, Cr, Cm

def trace_thermodynamic_variables(Psi, H0_inlet, R, omega, gamma, cp, R_gas):
    """
    Trace thermodynamic variables along streamlines and update their values.

    Args:
        Psi: Stream function array (10xN).
        H0_inlet: Total enthalpy at the inlet (scalar or array).
        R: Radial coordinate array (10xN).
        omega: Rotational speed (rad/s).
        gamma: Specific heat ratio.
        cp: Specific heat capacity (J/kg.K).
        R_gas: Specific gas constant (J/kg.K).

    Returns:
        Dict of updated thermodynamic properties.
    """
    H01_rel = np.zeros_like(Psi)
    H0_rel = np.zeros_like(Psi)
    

    S = np.zeros_like(Psi)
    beta = np.zeros_like(Psi)

    for i in range(1, N-1):  # Loop over nodes axially
        for j in range(1, M-1):  # Loop over nodes radially
            # Identify streamline origin using Psi
            a = (Psi[j, i] - Psi[j - 1, i]) / (Psi[j, i - 1] - Psi[j - 1, i - 1])
            b = 1 - a

            # Total enthalpy at streamline origin
            H01_rel[j, i] = a * H0_rel[j - 1, i] + b * H0_rel[j - 1, i - 1]
            
            # Local relative total enthalpy
            H0_rel[j, i] = H01_rel[j, i] - 0.5 * (a * omega[j, i])**2 + 0.5 * (R[j,i] * omega[j, i])**2
            
            # Calculate velocity components
            
            Cx, Cr, Cm = calculate_velocities(Psi, m_dot, R, rho)
            beta[j, i] = np.arctan(Cr[j, i] / Cx[j, i])
            
            # Calculate thermodynamic properties
            h = H0_rel[j, i] - 0.5 * Cm[j, i]**2
            
            T[j, i] = h / cp
            
            p[j, i] = p[j - 1, i] * (T[j, i] / T[j - 1, i]) ** (gamma / (gamma - 1))
            
            rho[j, i] = cp * p[j, i] / (R_gas * h)
            S[j, i] = cp * np.log(h / H0_inlet) - R_gas * np.log(p[j, i] / P0_inlet)
        
    return {"H0_rel": H0_rel, "Cx": Cx, "Cr": Cr, "Cm": Cm, "beta": beta, "T": T, "p": p, "rho": rho, "s": S}

# Convergence Check
def check_convergence(Psi,tolerance, max_iterations=iter_max):
    for iteration in range(max_iterations):
        Psi_old = Psi.copy()
        omega = calculate_vorticity(Psi, rC_theta, R, T, S, H0)
        Psi = update_stream_function(Psi, rho, R, omega, dx=0.01)
        Cx, Cr, Cm = calculate_velocities(Psi, m_dot, R, rho)
        thermodynamic_results = trace_thermodynamic_variables(Psi, H0_inlet, R, omega, gamma=1.4, cp=Cp, R_gas=R_gas)

        if np.max(np.abs(Psi - Psi_old)) <= tolerance:
            print(f"Converged after {iteration + 1} iterations.")
            break
    else:
        print("Did not converge.")

    return Psi, Cx, Cr, Cm

Psi, rho, p, T, H0_inlet, H0, m_dot, S = initialize_variables()
rC_theta = start_rotor(delta_rC_theta_TE_hub, delta_rC_theta_TE_tip, Psi)
C_theta = rC_theta / R

# Run Simulation
Psi, Cx, Cr,Cm = check_convergence(Psi, tolerance=1e-5)


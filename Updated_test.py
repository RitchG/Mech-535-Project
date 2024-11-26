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
p_inlet = 1e5  # Pa
Cp = 1005  # J/(kg*K)
Cx_inlet = 136  # m/s
N_rpm = 6000  # rpm
delta_rC_theta_TE_hub = 82.3  # m^2/s
delta_rC_theta_TE_tip = 84.4  # m^2/s
tolerance = 1e-5
alpha_inlet = np.deg2rad(30)  # radians
N = 3 * num_stations
M = num_stations
LE = num_stations  # Index for the leading edge
TE = LE + num_stations  # Index for the trailing edge

# Grid Generation
def generate_grid(blade_width, num_stations):
    x = np.linspace(-1.5, 1.5, N)
    r = np.linspace(r_hub, r_tip, M)
    return np.meshgrid(x, r)

X, R = generate_grid(blade_width, num_stations)

# Initialization
def initialize_variables(rho_inlet, T_inlet, Cp, Cx_inlet, X, R):
    Psi = (R**2 - 0.9**2) / (1.0**2 - 0.9**2)
    rho = np.full_like(Psi, rho_inlet)
    m_dot = rho_inlet * Cx_inlet * np.pi * (1.0**2 - 0.9**2)
    
    T = np.full_like(Psi, T_inlet)

    H0_inlet = Cp * T_inlet + 0.5 * Cx_inlet**2
    H0 = np.full_like(Psi, rho_inlet)

    S = np.zeros_like(Psi)  # Initialize entropy
    return Psi, rho, m_dot, H0, T, H0_inlet, S

Psi_initial, rho, m_dot, H0, T, H0_inlet, S = initialize_variables(rho_inlet, T_inlet, Cp, Cx_inlet, X, R)

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
    
    #Check Swirl has been distributed properly
    #print(rC[0,TE]-rC[0,TE-1])        
    
    # Update rC after the trailing edge
    for i in range(TE+1, N):
        rC[:,i] = rC[:, TE]

    return rC

rC_theta = start_rotor(delta_rC_theta_TE_hub, delta_rC_theta_TE_tip, Psi_initial)
C_theta = (rC_theta / R)


# Finging T, S and H0
def update_thermodynamics(Cx, Cr):
    T = np.zeros_like(Cx)
    S = np.zeros_like(Cx)
    H0 = np.zeros_like(Cx)
    p = np.zeros_like(Cx)

    for j in range(Cx.shape[0]):
        for i in range(Cx.shape[1]):
            C_local = np.sqrt(np.clip(Cx[j, i], -1e6, 1e6)**2 + np.clip(Cr[j, i], -1e6, 1e6)**2)

            H0[j, i] = H0_inlet - 0.5 * C_local**2
            T[j, i] = H0[j, i] / Cp

            p[j, i] = rho[j, i] * R_gas * T[j, i]

            if T[j, i] > 0 and p[j, i] > 0:
                S[j, i] = Cp * np.log(T[j, i] / T_inlet) - R_gas * np.log(p[j, i] / p_inlet)
    return T, S, H0

def calculate_vorticity(Psi, rC_theta,Cx, R, T, H0): #How to get T and H0?
    omega = np.zeros_like(Psi)
    S = np.zeros_like(Psi)
    for i in range(1, N):
        for j in range(1, M-1):
            # Calculate vorticity using finite differences and interpolated values
            term1 = rC_theta[j + 1, i] - rC_theta[j - 1, i]
            term2 = S[j + 1, i] - S[j - 1, i] 
            term3 = H0[j + 1, i] - H0[j - 1, i] 
            
            omega[j, i] = (np.pi / (Cx_inlet * (R[j,i]-R[j-1,i]))) * (((C_theta[j, i] / R[j, i]) *  term1) + T[j,i] * term2 - term3)
    return omega

def update_stream_function(Psi, rho, R, omega, blade_width):
    new_Psi = Psi.copy()
    for j in range(1, N-1):
        for i in range(1, M -1):
            # Coefficients for finite-difference method
            A = 1/(1 / (rho[i + 1, j] * R[i + 1, j]) +
                 1 / (rho[i - 1, j] * R[i - 1, j]) +
                 1 / (rho[i, j + 1] * R[i, j + 1]) +
                 1 / (rho[i, j - 1] * R[i, j - 1]))
            B = (Psi[i + 1, j] / (rho[i + 1, j] * R[i + 1, j]) +
                 Psi[i - 1, j] / (rho[i - 1, j] * R[i - 1, j]) +
                 Psi[i, j + 1] / (rho[i, j + 1] * R[i, j + 1]) +
                 Psi[i, j - 1] / (rho[i, j - 1] * R[i, j - 1]))
            new_Psi[i, j] = A * (B + ((X[i, j] - X[i, j -1])**2 * omega[i, j]))

    return new_Psi

def calculate_velocities(Psi, m_dot, R, rho):
    Cx = np.zeros_like(Psi)
    Cr = np.zeros_like(Psi)
    for i in range(1, N-1):
        for j in range(M-1):
            Cx[j, i] = m_dot / (2 * np.pi * R[j, i] * rho[j, i]) * (Psi[j + 1, i] - Psi[j - 1, i]) / (2 * blade_width)
            Cr[j, i] = -m_dot / (2 * np.pi * R[j, i] * rho[j, i]) * (Psi[j, i + 1] - Psi[j, i - 1]) / (2 * blade_width)
    
    return Cx, Cr

def check_convergence(Psi, tolerance, rho, R, m_dot, blade_width, H0, X):
    for iteration in range(500):
        Psi_old = Psi.copy()
        #print(f"Iteration {iteration + 1}", Psi)
        # Step 1: Calculate Vorticity
        Cx, Cr = calculate_velocities(Psi, m_dot, R, rho)
        omega = calculate_vorticity(Psi, rC_theta, Cx, R, T, H0)
        
        # Step 2: Update Stream Function
        Psi = update_stream_function(Psi, rho, R, omega, blade_width)
        
        # Step 3: Calculate Velocities
        Cx, Cr = calculate_velocities(Psi, m_dot, R, rho)
        
        # Check for convergence
        if np.max(np.abs(Psi - Psi_old)) <= tolerance:
            print(f"Converged after {iteration + 1} iterations.")
            break
    else:
        print("Did not converge within the maximum number of iterations.")
        print(f"Maximum residual: {np.max(np.abs(Psi - Psi_old))}")
    return Psi, Cx, Cr

def trace_thermodynamic_variables(H0_inlet, Cx, Cr):
    H0_rel = np.zeros_like(Cx)
    for i in range(1, N):
        for j in range(1, M): 
            H0_rel[j, i] = H0_inlet - 0.5 * (Cx[j, i]**2 + Cr[j, i]**2)
    return H0_rel

def calculate_results(Cx, Cr, H0_inlet, H0_rel):
    TE_radial_velocity = Cr[:, -1]
    LE_incidence = np.arctan2(Cr[:, 0], Cx[:, 0]) - alpha_inlet
    alpha_LE = np.arctan2(Cr[:, 0], Cx[:, 0])
    alpha_TE = np.arctan2(Cr[:, -1], Cx[:, -1])
    turning_deflection = alpha_TE - alpha_LE
    delta_H0 = H0_rel - H0_inlet
    pressure_rise = delta_H0 * rho_inlet * Cp
    reaction = (H0_rel - H0_inlet) / H0_inlet
    power_absorbed = m_dot * delta_H0
    return TE_radial_velocity, np.rad2deg(LE_incidence), np.rad2deg(turning_deflection), pressure_rise, reaction, power_absorbed

Psi, Cx, Cr = check_convergence(Psi_initial, tolerance, rho, R, m_dot, blade_width, H0, X)
H0_rel = trace_thermodynamic_variables(H0_inlet, Cx, Cr)
results = calculate_results(Cx, Cr, H0, H0_rel)

#print("Results:", results)


# Plotting the 3D blade shapre 
def blade_shape(X, R, rC_theta):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, R, rC_theta, cmap='viridis', edgecolor='none')
    ax.set_title("3D Blade Shape")
    ax.set_xlabel("Axial Position (X)")
    ax.set_ylabel("Radial Position (R)")
    ax.set_zlabel("Swirl (rC_theta)")
    ax.set_xlim(0, 0.5)
    plt.show()

# Plotting the axial velocity vs Radius
def axial_velocity(Cx, R):
    plt.figure(figsize=(8, 5))
    for i in range(Cx.shape[1]):  
        plt.plot(R[:, 0], Cx[:, i], label=f"X = {i}")
    plt.title("Axial Velocity (Cx) vs Radius")
    plt.xlabel("Radius (R)")
    plt.ylabel("Axial Velocity (Cx)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Plotting the radial velocity vs Radius
def radial_velocity(Cr, R):
    plt.figure(figsize=(8, 5))
    for i in range(Cr.shape[1]):  
        plt.plot(R[:, 0], Cr[:, i], label=f"X = {i}")
    plt.title("Radial Velocity (Cr) vs Radius")
    plt.xlabel("Radius (R)")
    plt.ylabel("Radial Velocity (Cr)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Plotting the tangential velocity vs Radius
def tangential_velocity(C_theta, R):
    plt.figure(figsize=(8, 5))
    for i in range(C_theta.shape[1]):  
        plt.plot(R[:, 0], C_theta[:, i], label=f"X = {i}")
    plt.title("Tangential Velocity (Cz) vs Radius")
    plt.xlabel("Radius (R)")
    plt.ylabel("Tangential Velocity (Cz)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Plotting the stream function vs Radius 
def psi(Psi, R):
    plt.figure(figsize=(8, 5))
    for i in range(Psi.shape[1]):  # Loop through axial positions
        plt.plot(R[:, 0], Psi[:, i], label=f"X = {i}")
    plt.title("Stream Function (\u03A8) from Hub to Shroud")
    plt.xlabel("Radius (R)")
    plt.ylabel("Stream Function (\u03A8)")
    plt.legend()
    plt.grid(True)
    plt.show()

blade_shape(X, R, rC_theta)  
axial_velocity(Cx, R)        
radial_velocity(Cr, R)      
tangential_velocity(C_theta, R)  
psi(Psi, R)
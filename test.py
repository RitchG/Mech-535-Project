import numpy as np

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
    H0_inlet = Cp * T_inlet + 0.5 * Cx_inlet**2
    return Psi, rho, m_dot, H0_inlet

Psi, rho, m_dot, H0_inlet = initialize_variables(rho_inlet, T_inlet, Cp, Cx_inlet, X, R)

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

rC_theta = start_rotor(delta_rC_theta_TE_hub, delta_rC_theta_TE_tip, Psi)
C_theta = (rC_theta / R)


def calculate_vorticity(Psi, rC_theta,Cx, R, S, T, H0):
    omega = np.zeros_like(Psi)
    for i in range(1, N):
        for j in range(1, M-1):
            # Calculate vorticity using finite differences and interpolated values
            term1 = rC_theta[j + 1, i] - rC_theta[j - 1, i]
            term2 = S[j + 1, i] - S[j - 1, i] # Notsure how to get S
            term3 = H0[j + 1, i] - H0[j - 1, i] # Notsure how to get H
            omega[j, i] = (np.pi / (Cx[j,i] * (R[j]-R[j-1]))) * (((C_theta[j, i] / R[j, i]) *  term1) + T[j,i] * term2 - term3)
    return omega

def update_stream_function(Psi, rho, R, omega, blade_width):
    new_Psi = Psi.copy()
    for i in range(1, num_stations - 1):
        for j in range(1, num_stations - 1):
            # Coefficients for finite-difference method
            A = (1 / (rho[i + 1, j] * R[i + 1, j]) +
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
    for i in range(1, num_stations - 1):
        for j in range(1, num_stations - 1):
            Cx[i, j] = m_dot / (2 * np.pi * R[i, j] * rho[i, j]) * (Psi[i, j + 1] - Psi[i, j - 1]) / (2 * blade_width)
            Cr[i, j] = -m_dot / (2 * np.pi * R[i, j] * rho[i, j]) * (Psi[i + 1, j] - Psi[i - 1, j]) / (2 * blade_width)
    return Cx, Cr

def check_convergence(Psi, tolerance, rho, R, m_dot, blade_width, H0_inlet, X):
    for iteration in range(50):
        Psi_old = Psi.copy()
        
        # Step 1: Calculate Vorticity
        Cx, Cr = calculate_velocities(Psi, m_dot, R, rho)
        omega = calculate_vorticity(Psi, rC_theta, Cx, R, S, T, H0)
        
        # Step 2: Update Stream Function
        Psi = update_stream_function(Psi, rho, R, omega, blade_width)
        
        # Step 3: Calculate Velocities
        Cx, Cr = calculate_velocities(Psi, m_dot, R, rho, X)
        
        # Check for convergence
        if np.max(np.abs(Psi - Psi_old)) < tolerance:
            print(f"Converged after {iteration + 1} iterations.")
            break
    else:
        print("Did not converge within the maximum number of iterations.")
        print(f"Maximum residual: {np.max(np.abs(Psi - Psi_old))}")
    return Psi, Cx, Cr

def trace_thermodynamic_variables(H0_inlet, Cx, Cr):
    H0_rel = np.zeros_like(Cx)
    for i in range(1, num_stations - 1):
        for j in range(1, num_stations - 1):
            H0_rel[i, j] = H0_inlet - 0.5 * (Cx[i, j]**2 + Cr[i, j]**2)
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
    return TE_radial_velocity, LE_incidence, turning_deflection, pressure_rise, reaction, power_absorbed

Psi, Cx, Cr = check_convergence(Psi, tolerance, rho, R, m_dot, blade_width, H0_inlet, X)
H0_rel = trace_thermodynamic_variables(H0_inlet, Cx, Cr)
results = calculate_results(Cx, Cr, H0_inlet, H0_rel)

#print("Results:", results)

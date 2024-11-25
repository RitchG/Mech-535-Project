import numpy as np

# Constants
blade_width = 0.1  # m
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

# Grid Generation
def generate_grid(blade_width, num_stations):
    x = np.linspace(-5 * blade_width, 5 * blade_width, 3 * num_stations)
    r = np.linspace(0.9, 1.0, num_stations)  # Hub to tip
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
def initialize_swirl(delta_rC_theta_TE_hub, delta_rC_theta_TE_tip, num_stations, Psi):
    delta_rC_theta_TE = np.linspace(delta_rC_theta_TE_hub, delta_rC_theta_TE_tip, num_stations)
    rC_theta = np.zeros_like(Psi)
    for i in range(1, num_stations):
        rC_theta[:, i] = rC_theta[:, i-1] + delta_rC_theta_TE * (i / num_stations)
    return rC_theta

rC_theta = initialize_swirl(delta_rC_theta_TE_hub, delta_rC_theta_TE_tip, num_stations, Psi)

def calculate_vorticity(Psi, rC_theta, Cx_inlet, R, X, T_inlet, H0_inlet):
    omega = np.zeros_like(Psi)
    for i in range(1, num_stations - 1):
        for j in range(1, num_stations - 1):
            # Calculate vorticity using finite differences and interpolated values
            term1 = rC_theta[i, j + 1] - rC_theta[i, j - 1]
            term2 = Psi[i, j + 1] - Psi[i, j - 1]
            omega[i, j] = (2 * np.pi / (2 * blade_width)) * (Cx_inlet * rC_theta[i, j] / R[i, j]) * term1 + T_inlet * term2 - H0_inlet
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

def calculate_velocities(Psi, m_dot, R, rho, X):
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
        omega = calculate_vorticity(Psi, rC_theta, Cx_inlet, R, X, T_inlet, H0_inlet)
        
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

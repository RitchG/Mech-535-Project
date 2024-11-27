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
incompressible = False
gamma = 1.4
cp = 1005



iter_num = 10 # Number of iterations for convergence
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
    
    # Update rC after the trailing edge
    for i in range(TE+1, N):
        rC[:,i] = rC[:, TE]

    return rC

rC_theta = start_rotor(delta_rC_theta_TE_hub, delta_rC_theta_TE_tip, Psi_initial)
C_theta = (rC_theta / R)


# Finging T, S, H0 and rho
def update_thermodynamics(Cx, Cr, incompressible):
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
            
            if incompressible == False:
                rho[j, i] = p[j, i] / (R_gas * T[j, i])
    return T, S, H0, rho

def calculate_vorticity(Psi, rC_theta,Cx, R, T, S, H0):
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
            print("rho :", rho[i + 1, j])
    return new_Psi


def calculate_velocities(Psi, m_dot, R, rho):
    Cx = np.zeros_like(Psi)
    Cr = np.zeros_like(Psi)

    dx = 0.01
    dr = 0.01

    for i in range(1, N - 1):
        for j in range(1, M - 1):
            # Calculate axial and radial velocities at internal points
            Cx[j, i] = m_dot / (2 * np.pi * R[j, i] * rho[j, i]) * (Psi[j + 1, i] - Psi[j - 1, i]) / (2 * dr)
            Cr[j, i] = -m_dot / (2 * np.pi * R[j, i] * rho[j, i]) * (Psi[j, i + 1] - Psi[j, i - 1]) / (2 * dx)

        # Set Cr = 0 at the walls
        Cr[0, i] = 0
        Cr[M - 1, i] = 0 

        # Calculate Cx at hub and shroud
        Cx[0, i] = m_dot / (2 * np.pi * R[0, i] * rho[0, i]) * (Psi[1, i] - Psi[0, i])/ (2 * dr)
        Cx[M - 1, i] = -m_dot / (2 * np.pi * R[M - 1, i] * rho[M - 1, i]) * (Psi[M - 1, i] - Psi[M - 2, i]) / (2 *dr)

    return Cx, Cr


def check_convergence(Psi, tolerance, rho, R, m_dot, blade_width, H0, X):
    for iteration in range(iter_num):
        Psi_old = Psi.copy()

        # Step 1: Calculate Vorticity
        if iteration == 0:
            Cx, Cr = calculate_velocities(Psi, m_dot, R, rho)
        
        #T, S, H0, rho = update_thermodynamics(Cx, Cr, incompressible)
        omega = calculate_vorticity(Psi, rC_theta, Cx, R, T, S, H0)
        
        # Step 2: Update Stream Function
        Psi = update_stream_function(Psi, rho, R, omega, 0.01)
        
        # Step 3: Calculate Velocities
        Cx, Cr = calculate_velocities(Psi, m_dot, R, rho)
        
        results = trace_thermodynamic_variables(Psi, H0_inlet, R, omega, gamma, cp, R_gas)
        #Psi = results["Psi"]
        Cx = results["Cx"]
        Cr = results["Cr"]
        rho = results["rho"]
        H0_rel = results["H0_rel"]

        # Check for convergence
        if np.max(np.abs(Psi - Psi_old)) <= tolerance:
            print(f"Converged after {iteration + 1} iterations.")
            break
        elif iteration == iter_num - 1:
            print(f"Did not converge after {iteration + 1} iterations.")
            print(f"Maximum residual: {np.max(np.abs(Psi - Psi_old))}")
    
    return Psi, Cx, Cr, omega, H0_rel

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
    H0_rel = np.zeros_like(Psi)
    p = np.zeros_like(Psi)
    rho = np.zeros_like(Psi)
    T = np.zeros_like(Psi)
    S = np.zeros_like(Psi)
    Cx = np.zeros_like(Psi)
    Cr = np.zeros_like(Psi)
    V = np.zeros_like(Psi)
    beta = np.zeros_like(Psi)

    for i in range(1, N):  # Loop over nodes axially
        for j in range(1, M):  # Loop over nodes radially
            # Identify streamline origin using Psi
            a = (Psi[j, i] - Psi[j - 1, i]) / (Psi[j, i - 1] - Psi[j - 1, i - 1])
            b = 1 - a

            # Total enthalpy at streamline origin
            H0_rel[j, i] = a * H0_rel[j - 1, i] + b * H0_rel[j - 1, i - 1]

            # Local relative total enthalpy
            H0_rel[j, i] -= 0.5 * omega[j, i]**2 * (R[j, i]**2 - R[j - 1, i]**2)

            # Calculate velocity components
            Cx[j, i] = R[j, i] * omega[j, i]
            Cr[j, i] = Cx[j, i] - (R[j, i] * omega[j, i])
            V[j, i] = np.sqrt(Cx[j, i]**2 + Cr[j, i]**2)
            beta[j, i] = np.arctan(Cr[j, i] / Cx[j, i])

            # Calculate thermodynamic properties
            h = H0_rel[j, i] - 0.5 * V[j, i]**2
            T[j, i] = h / cp
            p[j, i] = p[j - 1, i] * (T[j, i] / T[j - 1, i]) ** (gamma / (gamma - 1))
            rho[j, i] = p[j, i] / (R_gas * T[j, i])
            S[j, i] = cp * np.log(h / H0_inlet) - R_gas * np.log(p[j, i] / p[j - 1, i])

    return {"H0_rel": H0_rel, "Cx": Cx, "Cr": Cr, "V": V, "beta": beta, "T": T, "p": p, "rho": rho, "s": S}


def calculate_results(Cx, Cr, H0_inlet, H0_rel, LE, TE):

    # Trailing Edge (T.E.) radial velocity (Cr at trailing edge)
    TE_radial_velocity = Cr[:, TE-1].reshape(-1, 1)

    # Leading Edge (L.E.) incidence (difference between flow angle and blade angle)
    LE_incidence = (np.arctan2(Cr[:, LE-1], Cx[:, LE-1]) - alpha_inlet).reshape(-1, 1)

    # Turning Deflection (difference in flow angle between L.E. and T.E.)
    alpha_LE = np.arctan2(Cr[:, LE-1], Cx[:, LE-1])
    alpha_TE = np.arctan2(Cr[:, TE-1], Cx[:, TE-1])
    turning_deflection = (alpha_TE - alpha_LE).reshape(-1, 1)

    # Change in stagnation enthalpy
    delta_H0 = (H0_rel[:, TE-1] - H0_inlet).reshape(-1, 1)

    # Static pressure rise (from stagnation enthalpy change)
    pressure_rise = (delta_H0 * rho_inlet * Cp).reshape(-1, 1)

    # Total pressure rise (assuming isentropic flow for approximation)
    total_pressure_rise = (rho_inlet * delta_H0).reshape(-1, 1)

    # Reaction (enthalpy change ratio)
    reaction = ((H0_rel[:, TE-1] - H0_inlet) / H0_inlet).reshape(-1, 1)

    # Power absorbed by the rotor
    power_absorbed = (m_dot * delta_H0).reshape(-1, 1)

    return TE_radial_velocity,np.rad2deg(LE_incidence),np.rad2deg(turning_deflection),pressure_rise,total_pressure_rise,reaction,power_absorbed

Psi, Cx, Cr, omega, H0_rel = check_convergence(Psi_initial, tolerance, rho, R, m_dot, blade_width, H0, X)
#final = trace_thermodynamic_variables(Psi, H0_inlet, R, omega, gamma, cp, R_gas)["H0_rel"]


#Results at Hub, Midspan and Shroud

TE_radial_velocity, LE_incidence, turning_deflection, pressure_rise, total_pressure_rise, reaction, power_absorbed = calculate_results(Cx, Cr, H0_inlet, H0_rel, LE, TE)

#Print Results
print("Hub Results:")
print("TE Radial Velocity:", TE_radial_velocity[0])
print("LE Incidence:", LE_incidence[0])
print("Turning Deflection:", turning_deflection[0])
print("Static Pressure Rise:", pressure_rise[0])
print("Total Pressure Rise:", total_pressure_rise[0])
print("Reaction:", reaction[0])
print("Power Absorbed:", power_absorbed[0])

print("\nMidspan Results:")
print("TE Radial Velocity:", TE_radial_velocity[M // 2])
print("LE Incidence:", LE_incidence[M // 2])
print("Turning Deflection:", turning_deflection[M // 2])
print("Static Pressure Rise:", pressure_rise[M // 2])
print("Total Pressure Rise:", total_pressure_rise[M // 2])
print("Reaction:", reaction[M // 2])
print("Power Absorbed:", power_absorbed[M // 2])

print("\nTip Results:")
print("TE Radial Velocity:", TE_radial_velocity[-1])
print("LE Incidence:", LE_incidence[-1])
print("Turning Deflection:", turning_deflection[-1])
print("Static Pressure Rise:", pressure_rise[-1])
print("Total Pressure Rise:", total_pressure_rise[-1])
print("Reaction:", reaction[M-1])
print("Power Absorbed:", power_absorbed[-1])

print(reaction)
# Plotting the 3D blade shapre 
def blade_shape(X, R, rC_theta):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, R, rC_theta, cmap='viridis', edgecolor='none')
    ax.set_title("3D Blade Shape")
    ax.set_xlabel("Axial Position (X)")
    ax.set_ylabel("Radial Position (R)")
    ax.set_zlabel("Swirl (rC_theta)")
    ax.set_xlim(-0.5, 0.5)
    

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
    

# Plotting the stream function vs Radius 
def psi(Psi, R):
    plt.figure(figsize=(8, 5))
    for i in range(Psi.shape[1]):  # Loop through axial positions
        plt.plot(R[:,0], Psi[:, i], label=f"X = {i}")
    plt.title("Stream Function (\u03A8) from Hub to Shroud")
    plt.xlabel("Radius (R)")
    plt.ylabel("Stream Function (\u03A8)")
    plt.legend()
    plt.grid(True)
    

blade_shape(X, R, rC_theta)  
axial_velocity(Cx, R)        
radial_velocity(Cr, R)      
tangential_velocity(C_theta, R)  
psi(Psi, R)
#plt.show()
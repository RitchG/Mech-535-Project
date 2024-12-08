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
rpm = 6000 * 2 * np.pi / 60  # rad/s
delta_rC_theta_TE_hub = 82.3  # m^2/s
delta_rC_theta_TE_tip = 84.4  # m^2/s

incompressible = True
iter_max = 1000
N = num_stations
M = num_stations
LE = 5
TE = LE + N

# Grid Generation
def generate_grid():
    x1 = np.linspace(-0.55, -0.05, LE)
    x2 = np.linspace(-0.05, 0.05, N)
    x3 = np.linspace(-0.05, 0.55, 5)
    x = np.concatenate((x1, x2, x3))
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
    loss_factor = np.zeros_like(Psi)
    for i in range(LE+1,TE):
        loss_factor[:,i] = 0.5 / (TE - LE)

        

    return Psi, rho, p, T, H0_inlet, H0, m_dot, S, loss_factor



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



def calculate_vorticity(Psi, rC_theta, R, T,  H0 , S):
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
    
    for i in range(1, N - 1):  # Loop over axial positions
        for j in range(1, M - 1):  # Loop over radial positions

            # Interpolate rho and R at half-grid points
            # p1h = plus 1 half
            # m1h = minus 1 half
            rho_i_jp1h = (rho[j + 1, i] + rho[j, i]) / 2
            rho_i_jm1h = (rho[j - 1, i] + rho[j, i]) / 2
            rho_ip1h_j = (rho[j, i + 1] + rho[j, i]) / 2
            rho_im1h_j = (rho[j, i - 1] + rho[j, i]) / 2

            R_i_jp1h = (R[j + 1, i] + R[j, i]) / 2
            R_i_jm1h = (R[j - 1, i] + R[j, i]) / 2
            R_ip1h_j = (R[j, i + 1] + R[j, i]) / 2
            R_im1h_j = (R[j, i - 1] + R[j, i]) / 2
            
            
            # Calculate A_ij using interpolated values
            A_ij = 1 / (
                  1 / (rho_ip1h_j * R_ip1h_j)
                + 1 / (rho_im1h_j * R_im1h_j)
                + 1 / (rho_i_jp1h * R_i_jp1h)
                + 1 / (rho_i_jm1h * R_i_jm1h)
            )
            
            # Calculate B_ij using interpolated values
            B_ij = (
                  Psi[j, i + 1] / (rho_ip1h_j * R_ip1h_j)
                + Psi[j, i - 1] / (rho_im1h_j * R_im1h_j)
                + Psi[j + 1, i] / (rho_i_jp1h * R_i_jp1h)
                + Psi[j - 1, i] / (rho_i_jm1h * R_i_jm1h)
            )
            
            # Update Psi
            new_Psi[j, i] = A_ij * (B_ij + dx**2 * omega[j, i])
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
    Cx[0, :] = m_dot / (2 * np.pi * R[0, :] * rho[0, :]) * (Psi[1, :] - Psi[0, :])/ (2 * dr)
    Cx[M - 1, :] = m_dot / (2 * np.pi * R[M - 1, :] * rho[M - 1, :]) * (Psi[M - 1, :] - Psi[M - 2, :]) / (2 *dr)

    # Set Cr = 0 at the walls
    Cr[0, :] = 0
    Cr[M - 1, :] = 0
    
    Cx[0, N-1:N] = Cx[0, N-2]

    return Cx, Cr, Cm

def trace_thermodynamic_variables(Psi, H0_inlet, R, gamma, cp, R_gas):
    """
    Trace thermodynamic variables along streamlines and enforce the rothalpy condition.

    Args:
        Psi: Stream function array (MxN).
        H0_inlet: Total enthalpy at the inlet (scalar or array).
        R: Radial coordinate array (MxN).
        omega: Rotational speed (rad/s).
        gamma: Specific heat ratio.
        cp: Specific heat capacity (J/kg.K).
        R_gas: Specific gas constant (J/kg.K).

    Returns:
        Dict of updated thermodynamic properties.
    """
    
    S = np.zeros_like(Psi)
    S1 = np.zeros_like(Psi)
    beta = np.zeros_like(Psi)
    T = np.zeros_like(Psi)
    p = np.full_like(Psi, P0_inlet)
    P01_rel = np.zeros_like(Psi)
    P02_rel = np.zeros_like(Psi)

    # Rothalpy constant
    I = H0 - rpm * rC_theta
    H01 = H0
    H02 = np.zeros_like(Psi)
    H02_rel = np.zeros_like(Psi)
    H0_rel = I + (rpm * R)**2/2
    H01_rel = H0
    h = np.zeros_like(Psi)
    
    for i in range(1, N-1):  # Axial loop
        
        for j in range(1, M - 1):  # Radial loop
            # Identify streamline origin
            a = (Psi[j, i] - Psi[j + 1, i - 1]) / (Psi[j, i - 1] - Psi[j + 1, i - 1])
            b = 1 - a
            

            # Enforce rothalpy conservation
            #H0rel is onlydefined if both interpolationstations are within the blade
            if (i > LE and i <= TE):
                H01_rel[j, i] = a * H01_rel[j , i - 1] + b * H01_rel[j + 1, i - 1]

                #Interpolate the Radius associated with the interpolated value of H01_rel and calculate H02_rel
                H02_rel[j, i] = H01_rel[j, i] - (a * rpm * R[j - 1, i] + b * rpm * R[j - 1, i - 1])**2 / 2  + (rpm * R[j, i])**2 / 2
                
                

            if i < LE or i > TE:
                H01_rel[j, i] = H0_rel[j, i]
                H02_rel[j, i] = H0_rel[j, i] + (rpm * R[j, i])**2 / 2



            # Calculate velocity components
            Cx, Cr, Cm = calculate_velocities(Psi, m_dot, R, rho)
            beta[j, i] = np.arctan(Cr[j, i] / Cx[j, i])

            # Update thermodynamic properties
            #H02[j, i] = H01[j, i] + rpm * rC_theta[j, i] - (rpm * a * rC_theta[j , i - 1] + b * rC_theta[j + 1, i - 1])
            #H02_rel[j, i] = I[j, i] + (rpm * R[j, i])**2 / 2
            h[j, i] = H02_rel[j, i] - 0.5 * (Cm[j, i]**2 + C_theta[j, i]**2)
            P01_rel[j, i] = p[j,i] * (H01_rel[j, i] / (I[j, i] + (rpm * rC_theta[j, i]))) ** (gamma / (gamma - 1))
            P02_rel[j, i] = P01_rel[j, i] * (H0_rel[j, i] / H01_rel[j, i]) ** (gamma / (gamma - 1))

            p[j, i] = P02_rel[j, i] - loss_factor[j, i] * (P02_rel[j, i] - p[j,i-1])
            
            if incompressible == False:
                T[j, i] = H0_rel[j, i] / cp
                rho[j, i] = p[j, i] / (R_gas * T[j, i])

                S1[j, i] = a * S[j , i - 1] + b * S[j + 1, i - 1]
                S[j, i] = S1[j,i] + cp * np.log(H0_rel[j, i] / H01_rel[j, i]) - R_gas * np.log(p[j, i] / p[j,i-1])
    return {"H0_rel": H0_rel, "Cx": Cx, "Cr": Cr, "beta": beta, "T": T, "p": p, "rho": rho, "S": S}

# Convergence Check
def check_convergence(Psi,tolerance, max_iterations=iter_max):
    for iteration in range(max_iterations):
        Psi_old = Psi.copy()
        if iteration == 0:
            S = np.zeros_like(Psi)
            omega = calculate_vorticity(Psi, rC_theta, R, T, H0, S)
        else:
            omega = calculate_vorticity(Psi, rC_theta, R, T, H0, S)
        Psi = update_stream_function(Psi, rho, R, omega, dx=0.01)
        Cx, Cr, Cm = calculate_velocities(Psi, m_dot, R, rho)
        thermodynamic_results = trace_thermodynamic_variables(Psi, H0_inlet, R, gamma=1.4, cp=Cp, R_gas=R_gas)
        S = thermodynamic_results["S"]
        if np.max(np.abs(Psi - Psi_old)) <= tolerance:
            print(f"Converged after {iteration + 1} iterations.")
            break
    else:
        print("Did not converge.")

    return Psi, Cx, Cr, Cm, thermodynamic_results, omega

Psi, rho, p, T, H0_inlet, H0, m_dot, S, loss_factor = initialize_variables()
rC_theta = start_rotor(delta_rC_theta_TE_hub, delta_rC_theta_TE_tip, Psi)
C_theta = rC_theta / R


# Run Simulation
Psi, Cx, Cr, Cm, thermodynamic_results, omega = check_convergence(Psi, tolerance=1e-5)


# Assuming Psi and R are already defined from the simulation
def plot_stream_function(Psi, R):
    plt.figure(figsize=(10, 6))
    plt.contourf(X, R, Psi, levels=50, cmap='viridis')  # Use sufficient levels
    plt.colorbar(label='Streamfunction')
    plt.xlabel('Axial Position')
    plt.ylabel('Radial Position')
    plt.title('Streamfunction Contours')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
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
    

def calculate_rotor_properties(data):
    """
    Calculates and prints rotor performance properties at the hub, midspan, and tip.

    Args:
        data (dict): Dictionary containing 10x20 arrays for:
            - "H0_rel": Relative total enthalpy.
            - "Cx": Axial velocity.
            - "Cr": Radial velocity.
            - "beta": Blade angle (radians).
            - "T": Temperature (K).
            - "p": Static pressure (Pa).
            - "rho": Density (kg/m^3).
            - "S": Entropy (J/kg.K).
    """
    radial_indices = [0, M // 2, M - 1]
    locations = ["Hub", "Midspan", "Tip"]

    results = {}

    for r_idx, loc in zip(radial_indices, locations):
        # Extract relevant values
        beta_LE = data["beta"][r_idx, LE]
        beta_TE = data["beta"][r_idx, TE]
        Cx_LE = data["Cx"][r_idx, LE]
        Cx_TE = data["Cx"][r_idx, TE]
        Cr_LE = data["Cr"][r_idx, LE]
        Cr_TE = data["Cr"][r_idx, TE]
        p_LE = data["p"][r_idx, LE]
        p_TE = data["p"][r_idx, TE]
        rho_LE = data["rho"][r_idx, LE]
        rho_TE = data["rho"][r_idx, TE]
        H0_rel_LE = data["H0_rel"][r_idx, LE]
        H0_rel_TE = data["H0_rel"][r_idx, TE]

        # Trailing Edge Radial Velocity
        Cr_TE_val = Cr_TE

        # Leading Edge Incidence
        incidence = np.rad2deg(beta_LE - np.arctan(Cr_LE / Cx_LE))

        # Turning (Deflection)
        deflection = np.rad2deg(beta_LE - beta_TE)

        # Static Pressure Rise
        delta_p = p_TE - p_LE

        # Total Pressure Rise
        C_TE = np.sqrt(Cx_TE**2 + Cr_TE**2)
        C_LE = np.sqrt(Cx_LE**2 + Cr_LE**2)
        total_pressure_rise = 0.5 * rho_TE * C_TE**2 - 0.5 * rho_LE * C_LE**2

        # Reaction
        static_enthalpy_TE = H0_rel_TE - 0.5 * C_TE**2
        reaction = static_enthalpy_TE / H0_rel_TE

        power_absorbed = m_dot * omega[r_idx,TE] * R[r_idx, TE] * (Cx_TE * np.tan(beta_TE) - Cx_LE * np.tan(beta_LE))

        # Store results
        results[loc] = {
            "Radial Velocity (Cr_TE)": Cr_TE_val,
            "Incidence (deg)": incidence,
            "Turning (Deflection, deg)": deflection,
            "Static Pressure Rise (Pa)": delta_p,
            "Total Pressure Rise (Pa)": total_pressure_rise,
            "Reaction": reaction,
            "Power Absorbed (W)": power_absorbed,
        }

    # Print results for each location
    for loc, props in results.items():
        print(f"\n{loc}:")
        for key, value in props.items():
            print(f"  {key}: {value:.2f}")

    return results

results = calculate_rotor_properties(thermodynamic_results)

plot_stream_function(Psi, R)
blade_shape(X, R, rC_theta)  
axial_velocity(Cx, R)        
radial_velocity(Cr, R)      
tangential_velocity(C_theta, R)  
psi(Psi, R)
plt.show()
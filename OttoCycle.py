#-----------------------------------------------------------------------------#
#=============================================================================#
#-----------------------------------------------------------------------------#

# PME-3480 - Motores de Combustão Interna
# 1D Otto cycle simulator - 2025
# Functions file

#-----------------------------------------------------------------------------#
#=============================================================================#
#-----------------------------------------------------------------------------#

# Importing libraries
import numpy as np  # NumPy is a library for numerical operations with multi-dimensional arrays and matrices, and provides mathematical functions.
import cantera as ct  # Cantera is an open-source suite of tools for chemical kinetics, thermodynamics, and transport processes.
from scipy.integrate import odeint  # Imports the `odeint` function from the `scipy.integrate` module, which is used to solve ordinary differential equations (ODEs).
from tqdm import tqdm  # Imports the `tqdm` module, which provides a progress bar to visually track the progress of loops and iterations in the code.

#-----------------------------------------------------------------------------#
#=============================================================================#
#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#
# Functions
#-----------------------------------------------------------------------------#

def volume(Th, pars):
    
    """
    Calculates the cylinder volume at different crank angles (Theta).
    
    Parameters:
    Th (numpy array): Array of crank angles in radians.
    pars (tuple): Tuple of engine parameters.
        case (str): Case identifier ('motored' or 'fired').
        B (float): Cylinder bore (m).
        S (float): Piston stroke (m).
        L (float): Connecting rod length (m).
        rv (float): Compression ratio
    
    Returns:
    tuple: (V, dVdTh)
        V (numpy array): Array of cylinder volumes in cubic meters (m³).
        dVdTh (numpy array): Array of derivative of cylinder volume with respect to crank angle (m³/rad).
    """

    #-------------------------------------------------------------------------#
    # Unpacking parameters
    #-------------------------------------------------------------------------#
    B = pars[1] # Cylinder bore - m
    S = pars[2] # Piston stroke - m 
    L = pars[3] # Connecting rod length - m
    rv = pars[4] # Compression ratio
    
    #-------------------------------------------------------------------------#
    # Crank radius - m
    #-------------------------------------------------------------------------#
    r = S/2

    #-------------------------------------------------------------------------#    
    # Piston position relative to TDC (Top Dead Center) - m
    #-------------------------------------------------------------------------#
    z = r*(1 - np.cos(Th)) + L*(1 - np.sqrt(1 - ((r/L)**2)*(np.sin(Th)**2)))
    
    #-------------------------------------------------------------------------#
    # Cylinder displacement volume - m³
    #-------------------------------------------------------------------------#
    Vu = (np.pi/4)*(B**2)*S
    
    #-------------------------------------------------------------------------#
    # Combustion chamber volume - m³
    #-------------------------------------------------------------------------#
    Vcam = Vu/(rv-1)
    
    #-------------------------------------------------------------------------#
    # Total cylinder volume - m³
    #-------------------------------------------------------------------------#
    V = Vcam + (np.pi/4)*(B**2)*z
    
    #-------------------------------------------------------------------------#
    # Derivative of cylinder volume with respect to crank angle - m³/rad
    #-------------------------------------------------------------------------#
    dzdTh = r*np.sin(Th) + ((r**2)/L)*np.sin(Th)*np.cos(Th)*((1 - ((r/L)**2)*(np.sin(Th)**2))**(-0.5))
    dVdTh = (np.pi/4)*(B**2)*dzdTh
    
    return V, dVdTh

#-----------------------------------------------------------------------------#

def intakeModel(Th, p, T, pars):

    """
    Defines the intake mass flow rate model as a function of crank angle.

    Parameters:
    Th (numpy array): Array of crank angles in radians.
    p (numpy array): Pressure at intake (Pa).
    T (numpy array): Temperature at intake (K).
    pars (tuple): Tuple of engine parameters.
        Th0 (float): Intake valve opening angle in radians.
        Th1 (float): Intake valve closing angle in radians.

    Returns:
    numpy array: Array of intake mass flow rates at the given crank angles (kg/s).
    """
    
    #-------------------------------------------------------------------------#
    # Unpacking parameters
    #-------------------------------------------------------------------------#
    case = pars[0]  # Engine case: 'motored' or 'fired'
    Th0 = pars[6]  # Intake valve opening angle in radians
    Th1 = pars[7]  # Intake valve closing angle in radians
    p1 = pars[14] # pressure - Pa
    T1 = pars[15] # temperature - K
    
    #-------------------------------------------------------------------------#    
    # Intake mass flow rate model - compressible flow
    #-------------------------------------------------------------------------#

    if case == 'motored':

        #---------------------------------------------------------------------#
        # air properties
        #---------------------------------------------------------------------#
        airProperties = np.load('./Files/properties-air.npy')
        R = airProperties[5] # gas constant - J/kgK
        k = airProperties[2] # specific heat ratio

    elif case == 'fired':

        #---------------------------------------------------------------------#
        # air properties
        #---------------------------------------------------------------------#
        unburntProperties = np.load('./Files/properties-unburnt.npy')
        R = unburntProperties[5] # gas constant - J/kgK
        k = unburntProperties[2] # specific heat ratio

    #-------------------------------------------------------------------------#
    # Stagnation conditions - 0
    #-------------------------------------------------------------------------#
    V1 = 0.
    M1 = V1/np.sqrt(k*R*T1)
    psi1 = 1+((k-1)/2)*(M1**2)
    p0 = p1*(psi1**(k/(k-1)))
    T0 = T1*psi1

    #-------------------------------------------------------------------------#
    # Intake valve parameters
    #-------------------------------------------------------------------------#
    db = 55/1000 # 50/1000 # Valve diameter - m
    Ab = (np.pi / 4) * (db**2)  # Valve area - m2
    dh = 3/1000  # valve stem diameter - m
    Ah = (np.pi / 4) * (dh**2)  # stem area - m2
    Av = Ab-Ah

    #-------------------------------------------------------------------------#
    # critical properties
    #-------------------------------------------------------------------------#
    pc = p0*((2/(k+1))**(k/(k-1))) # pressure - Pa
    Ac = Av

    #-------------------------------------------------------------------------#
    # Define the intake mass flow rate
    #-------------------------------------------------------------------------#
    M = np.sqrt(np.clip((2/(k-1))*(((p0/p)**((k-1)/k))-1), 0, None))
    mdots = Av*p0*np.sqrt((k/(R*T0)))*M*((1+((k-1)/2)*(M**2))**(-(k+1)/(2*(k-1)))) # mass flow rate
    mdotc = Ac*p0*np.sqrt((k/(R*T0)))*((2/(k+1))**((k+1)/(2*(k-1)))) # critical mass flow rate
    mdot = np.where(p>pc, mdots, mdotc)
    
    mdotInt = np.zeros_like(Th)
    condition_open = (Th >= Th0) | (Th <= Th1)
    mdotInt[condition_open] = mdot[condition_open]
    
    return mdotInt

#-----------------------------------------------------------------------------#

def exhaustModel(Th, p, T, pars):
    
    """
    Defines the exhaust mass flow rate model as a function of crank angle.

    Parameters:
    Th (numpy array): Array of crank angles in radians.
    p (numpy array): Array of pressures at exhaust (Pa).
    T (numpy array): Array of temperatures at exhaust (K).
    pars (tuple): Tuple of engine parameters.
        Th0 (float): Exhaust valve opening angle in radians.
        Th1 (float): Exhaust valve closing angle in radians.

    Returns:
    numpy array: Array of exhaust mass flow rates at the given crank angles (kg/s).
    """
    
    #-------------------------------------------------------------------------#
    # Unpacking parameters
    #-------------------------------------------------------------------------#
    case = pars[0]  # Engine case: 'motored' or 'fired'
    Th0 = pars[8]  # Exhaust valve opening angle in radians
    Th1 = pars[9]  # Exhaust valve closing angle in radians
    p2 = pars[16] # pressure - Pa

    #-------------------------------------------------------------------------#    
    # Intake mass flow rate model - compressible flow
    #-------------------------------------------------------------------------#

    if case == 'motored':

        #---------------------------------------------------------------------#
        # air properties
        #---------------------------------------------------------------------#
        airProperties = np.load('./Files/properties-air.npy')
        R = airProperties[5] # gas constant - J/kgK
        k = airProperties[2] # specific heat ratio

    elif case == 'fired':

        #---------------------------------------------------------------------#
        # air properties
        #---------------------------------------------------------------------#
        burntProperties = np.load('./Files/properties-unburnt.npy')
        R = burntProperties[5] # gas constant - J/kgK
        k = burntProperties[2] # specific heat ratio

    #-------------------------------------------------------------------------#
    # Stagnation conditions - 0
    #-------------------------------------------------------------------------#
    V1 = 0.
    M1 = V1/np.sqrt(k*R*T)
    psi1 = 1+((k-1)/2)*(M1**2)
    p0 = p*(psi1**(k/(k-1)))
    T0 = T*psi1

    #-------------------------------------------------------------------------#
    # Intake valve parameters
    #-------------------------------------------------------------------------#
    db = 35/1000 # Valve diameter - m
    Ab = (np.pi / 4) * (db**2)  # Valve area - m2
    dh = 3/1000  # valve stem diameter - m
    Ah = (np.pi / 4) * (dh**2)  # stem area - m2
    Av = Ab-Ah

    #-------------------------------------------------------------------------#
    # critical properties
    #-------------------------------------------------------------------------#
    pc = p0*((2/(k+1))**(k/(k-1))) # pressure - Pa
    Ac = Av

    #-------------------------------------------------------------------------#
    # Define the exhaust mass flow rate
    #-------------------------------------------------------------------------#
    M = np.sqrt(np.clip((2/(k-1))*(((p0/p2)**((k-1)/k))-1), 0, None))
    mdots = Av*p0*np.sqrt((k/(R*T0)))*M*((1+((k-1)/2)*(M**2))**(-(k+1)/(2*(k-1)))) # mass flow rate
    mdotc = Ac*p0*np.sqrt((k/(R*T0)))*((2/(k+1))**((k+1)/(2*(k-1)))) # critical mass flow rate
    mdot = np.where(p>pc, mdots, mdotc)
    
    mdotExh = np.zeros_like(Th)
    condition_open = (Th <= Th1) | (Th >= Th0)
    mdotExh[condition_open] = mdot[condition_open]
    
    return mdotExh

#-----------------------------------------------------------------------------#

def massODE(Th, m, p, T, pars):
    
    """
    Defines the ODE for mass as a function of crank angle.

    Parameters:
    Th (numpy array): Array of crank angles in radians.
    m (numpy array): Array of masses at given crank angles (kg).
    p (numpy array): Array of pressures (Pa).
    T (numpy array): Array of temperatures (K).
    pars (tuple): Tuple of engine parameters.
        n (float): Engine speed in revolutions per second (rps).

    Returns:
    numpy array: Array of rates of change of mass with respect to crank angle (kg/rad).
    """

    #-------------------------------------------------------------------------#
    # Unpacking parameters
    #-------------------------------------------------------------------------#
    n = pars[5] # Engine speed in rps
    
    #-------------------------------------------------------------------------#
    # Compute intake and exhaust mass flow rates at crank angle Th
    #-------------------------------------------------------------------------#
    mdotInt = intakeModel(Th, p, T, pars) # Intake mass flow rate - kg/s
    mdotExh = exhaustModel(Th, p, T, pars) # Exhaust mass flow rate - kg/s

    #-------------------------------------------------------------------------#
    # Net mass flow rate into the cylinder
    #-------------------------------------------------------------------------#
    mdot = (mdotInt - mdotExh)/(2*np.pi*n)
    
    return mdot

#-----------------------------------------------------------------------------#

def wiebeFunction(Th, pars):
    
    """
    Calculates the Wiebe function.
    
    Parameters:
    Th (numpy array): Array of crank angles in radians.
    pars (tuple): Tuple of engine parameters.
        ThSOC (float): Start of combustion angle in radians.
        ThEOC (float): End of combustion angle in radians.
        a (float): Wiebe efficiency factor.
        m (float): Wiebe form factor (m+1).

    Returns:
    tuple: (xb, dxb)
        xb (numpy array): Array of Wiebe function values corresponding to the crank angles.
        dxb (numpy array): Array of derivatives of the Wiebe function with respect to crank angle.
    """
    
    #-------------------------------------------------------------------------#
    # Unpacking parameters
    #-------------------------------------------------------------------------#
    ThSOC = pars[10] # Start of combustion angle (converted to radians)
    ThEOC = pars[11] # End of combustion angle (converted to radians)
    a = pars[12] # Wiebe efficiency factor
    m = pars[13] # Wiebe form factor (m+1)

    #-------------------------------------------------------------------------#
    # Piecewise function to define the Wiebe function and its derivative
    #-------------------------------------------------------------------------#
    xb = np.piecewise(Th, [Th<ThSOC, Th>=ThSOC, Th>=ThEOC],
                        [lambda Th: 0,
                        lambda Th: 1-np.exp(-a*(((Th-ThSOC)/(ThEOC-ThSOC))**(m+1))),
                        lambda Th: 1])

    dxb = np.piecewise(Th, [Th<ThSOC, Th>=ThSOC, Th>=ThEOC],
                        [lambda Th: 0,
                        lambda Th: (a*(m+1)*(((Th-ThSOC)/(ThEOC-ThSOC))**m)/(ThEOC-ThSOC))*np.exp(-a*(((Th-ThSOC)/(ThEOC-ThSOC))**(m+1))),
                        lambda Th: 0])
    
    return xb, dxb

#-----------------------------------------------------------------------------#

def heatTermSolver(Th, m, p, pars):
    
    """
    Solves the heat term in the temperature ODE based on energy balance.
    
    Parameters:
    Th (numpy array): Array of crank angles in radians.
    m (numpy array): Array of masses at corresponding crank angles (kg).
    p (numpy array): Array of pressures at corresponding crank angles (Pa).
    pars (tuple): Tuple of engine parameters.
        case (str): Engine case: 'motored' or 'fired'.
        B (float): Cylinder bore (m).
        S (float): Piston stroke (m).
        L (float): Connecting rod length (m).
        rv (float): Compression ratio.
        n (float): Engine speed in revolutions per second (rps).
        initial temperature (float): Initial temperature (K).

    Returns:
    numpy array: Array of heat terms corresponding to the crank angles (J/kg).
    """

    #-------------------------------------------------------------------------#
    # Unpacking parameters
    #-------------------------------------------------------------------------#
    case = pars[0]  # Engine case: 'motored' or 'fired'
    n = pars[7]     # Engine speed - rps
    
    #-------------------------------------------------------------------------#
    # Gas properties
    #-------------------------------------------------------------------------#
    
    #-------------------------------------------------------------------------#
    # Properties - air or Unburnt/Burnt (Wiebe Function)
    #-------------------------------------------------------------------------#
    if case == 'motored':

        #---------------------------------------------------------------------#
        # air properties
        #---------------------------------------------------------------------#
        airProperties = np.load('./Files/properties-air.npy')

        #---------------------------------------------------------------------#
        # Specific heat capacity - J/kgK
        #---------------------------------------------------------------------#
        cv = airProperties[3]

    elif case == 'fired':

        #---------------------------------------------------------------------#
        # Mixture - Unburnt/Burnt
        #---------------------------------------------------------------------#

        #---------------------------------------------------------------------#
        # Unburnt State
        #---------------------------------------------------------------------#
        # Load unburnt state properties from file
        unburntProperties = np.load('./Files/properties-unburnt.npy')
        
        #---------------------------------------------------------------------#
        # Burnt State
        #---------------------------------------------------------------------#
        # Load burnt state properties from file
        burntProperties = np.load('./Files/properties-burnt.npy')
        
        #---------------------------------------------------------------------#
        # Wiebe function
        #---------------------------------------------------------------------#
        xb, _ = wiebeFunction(Th, pars) # Calculate burnt mass fraction using Wiebe function
        xu = 1-xb # unburnt mass fraction

        #---------------------------------------------------------------------#
        # Specific heat - J/kgK
        #---------------------------------------------------------------------#
        cvu = unburntProperties[3] # unburnt constant-volume specific heat
        cvb = burntProperties[3] # burnt constant-volume specific heat

        cv = xu*cvu + xb*cvb # mixture constant-volume specific heat
    
    #-------------------------------------------------------------------------#
    # Heat solver
    #-------------------------------------------------------------------------#
    
    #-------------------------------------------------------------------------#
    # Placeholder for heat transfer rate (to be implemented based on specific model)
    #-------------------------------------------------------------------------#
    dQdt = -np.zeros_like(Th)  # Placeholder, needs to be implemented based on the actual model
    
    #-------------------------------------------------------------------------#
    # Heat term calculation
    #-------------------------------------------------------------------------#
    ht = dQdt/(m*cv*2*np.pi*n)
    
    return ht

#-----------------------------------------------------------------------------#

def workTermSolver(Th, V, dV, pars):
    
    """
    Solves the work term in the temperature ODE based on cylinder volume derivative.
    
    Parameters:
    Th (numpy array): Array of crank angles in radians.
    V (numpy array): Array of cylinder volumes at corresponding crank angles (m³).
    dV (numpy array): Array of derivatives of cylinder volumes with respect to crank angles (m³/rad).
    pars (tuple): Tuple of engine parameters.
        case (str): Engine case: 'motored' or 'fired'.
        B (float): Cylinder bore (m).
        S (float): Piston stroke (m).
        L (float): Connecting rod length (m).
        rv (float): Compression ratio.
        other parameters (float): Additional parameters as needed.
        initial temperature (float): Initial temperature (K).

    Returns:
    numpy array: Array of work terms corresponding to the crank angles (J/kg).
    """
    
    #-------------------------------------------------------------------------#
    # Unpacking parameters
    #-------------------------------------------------------------------------#
    case = pars[0]  # Engine case: 'motored' or 'fired'
    
    #-------------------------------------------------------------------------#
    # Gas properties
    #-------------------------------------------------------------------------#
    
    #-------------------------------------------------------------------------#
    # Properties - air or Unburnt/Burnt (Wiebe Function)
    #-------------------------------------------------------------------------#

    if case == 'motored':

        #---------------------------------------------------------------------#
        # air properties
        #---------------------------------------------------------------------#
        airProperties = np.load('./Files/properties-air.npy')

        #---------------------------------------------------------------------#
        # Specific heat capacity - J/kgK
        #---------------------------------------------------------------------#
        cv = airProperties[3]

        #---------------------------------------------------------------------#
        # Molar mass - kg/kmol
        #---------------------------------------------------------------------#
        M = airProperties[6]
        
        #---------------------------------------------------------------------#
        # Ideal gas constant - J/kmol*K
        #---------------------------------------------------------------------#
        RU = ct.gas_constant
        
        #---------------------------------------------------------------------#
        # Gas constant - J/kgK
        #---------------------------------------------------------------------#
        R = RU/M
        
    elif case == 'fired':

        #---------------------------------------------------------------------#
        # Mixture - Unburnt/Burnt
        #---------------------------------------------------------------------#

        #---------------------------------------------------------------------#
        # Unburnt State
        #---------------------------------------------------------------------#
        # Load unburnt state properties from file
        unburntProperties = np.load('./Files/properties-unburnt.npy')
        
        #---------------------------------------------------------------------#
        # Burnt State
        #---------------------------------------------------------------------#
        # Load burnt state properties from file
        burntProperties = np.load('./Files/properties-burnt.npy')

        #---------------------------------------------------------------------#
        # Wiebe function
        #---------------------------------------------------------------------#
        xb, _ = wiebeFunction(Th, pars) # Calculate burnt mass fraction using Wiebe function
        xu = 1-xb # unburnt mass fraction

        #---------------------------------------------------------------------#
        # Specific heat - J/kgK
        #---------------------------------------------------------------------#
        cvu = unburntProperties[3] # unburnt constant-volume specific heat
        cvb = burntProperties[3] # burnt constant-volume specific heat

        cv = xu*cvu + xb*cvb # mixture constant-volume specific heat

        #---------------------------------------------------------------------#
        # Molar mass - kg/kmol
        #---------------------------------------------------------------------#
        Mu = unburntProperties[6] # unburnt molar mass
        Mb = burntProperties[6] # burnt molar mass
        M = 1/(xu/Mu + xb/Mb) # molar mass of the mixture
        
        #---------------------------------------------------------------------#
        # Ideal gas constant - J/kmol*K
        #---------------------------------------------------------------------#
        RU = ct.gas_constant
        
        #---------------------------------------------------------------------#
        # Gas constant - J/kgK
        #---------------------------------------------------------------------#
        R = RU/M
    
    #-------------------------------------------------------------------------#
    # Work term calculation
    #-------------------------------------------------------------------------#
    wt = (R/cv)*(dV/V)
    
    return wt

#-----------------------------------------------------------------------------#

def reactionTermSolver(Th, pars):
    
    """
    Solves the reaction term in the temperature ODE based on the Wiebe function derivative.
    
    Parameters:
    Th (numpy array): Array of crank angles in radians.
    pars (tuple): Tuple of engine parameters.
        case (str): Engine case: 'motored' or 'fired'.
        B (float): Cylinder bore (m).
        S (float): Piston stroke (m).
        L (float): Connecting rod length (m).
        rv (float): Compression ratio.
        other parameters (float): Additional parameters as needed.
        initial temperature (float): Initial temperature (K).

    Returns:
    numpy array: Array of reaction terms corresponding to the crank angles (J/kg).
    """
    
    #-------------------------------------------------------------------------#    
    # Unpacking parameters
    #-------------------------------------------------------------------------#
    case = pars[0]  # Engine case: 'motored' or 'fired'
    
    #-------------------------------------------------------------------------#
    # Gas properties
    #-------------------------------------------------------------------------#
    
    #-------------------------------------------------------------------------#
    # Properties - air or Unburnt/Burnt (Wiebe Function)
    #-------------------------------------------------------------------------#
    
    if case == 'motored':

        #---------------------------------------------------------------------#
        # air properties
        #---------------------------------------------------------------------#
        airProperties = np.load('./Files/properties-air.npy')

        #---------------------------------------------------------------------#
        # specific heat - J/kgK
        #---------------------------------------------------------------------#
        cv = airProperties[3] # constant-volume specific heat

        #---------------------------------------------------------------------#
        # internal energy - J/kg
        #---------------------------------------------------------------------#
        uf = airProperties[7] # formation
        ufb, ufu = uf, uf
        
        #---------------------------------------------------------------------#
        # For motored case, no combustion reaction occurs
        #---------------------------------------------------------------------#
        dxbdTh = 0

    elif case == 'fired':

        #---------------------------------------------------------------------#
        # Mixture - Unburnt/Burnt
        #---------------------------------------------------------------------#

        #---------------------------------------------------------------------#
        # Unburnt State
        #---------------------------------------------------------------------#
        # Load unburnt state properties from file
        unburntProperties = np.load('./Files/properties-unburnt.npy')
        
        #---------------------------------------------------------------------#
        # Burnt State
        #---------------------------------------------------------------------#
        # Load burnt state properties from file
        burntProperties = np.load('./Files/properties-burnt.npy')

        #---------------------------------------------------------------------#
        # Wiebe function
        #---------------------------------------------------------------------#
        xb, dxbdTh = wiebeFunction(Th, pars) # Calculate burnt mass fraction using Wiebe function
        xu = 1-xb # unburnt mass fraction
        
        #---------------------------------------------------------------------#
        # Specific heat - J/kgK
        #---------------------------------------------------------------------#
        cvu = unburntProperties[3] # unburnt constant-volume specific heat
        cvb = burntProperties[3] # burnt constant-volume specific heat
        
        cv = xu*cvu + xb*cvb # mixture constant-volume specific heat
        
        #---------------------------------------------------------------------#
        # internal energy - J/kg
        #---------------------------------------------------------------------#
        ufu = unburntProperties[7] # unburnt formation
        ufb = burntProperties[7] # burnt formation
        
    #-------------------------------------------------------------------------#
    # Reaction term solver
    #-------------------------------------------------------------------------#
    
    #-------------------------------------------------------------------------#
    # Calculate reaction term based on Wiebe function derivative
    #-------------------------------------------------------------------------#
    rt = dxbdTh*((ufb - ufu)/cv)
    
    return rt

#-----------------------------------------------------------------------------#

def massTermSolver(Th, m, p, T, pars):
    
    """
    Solves the mass term in the temperature ODE based on intake and exhaust models.
    
    Parameters:
    Th (numpy array): Array of crank angles in radians.
    m (numpy array): Array of masses at the corresponding crank angles.
    p (numpy array): Array of pressures at the corresponding crank angles.
    T (numpy array): Array of temperatures at the corresponding crank angles.
    pars (tuple): Tuple of engine parameters.
        case (str): Engine case: 'motored' or 'fired'.
        B (float): Cylinder bore (m).
        S (float): Piston stroke (m).
        L (float): Connecting rod length (m).
        rv (float): Compression ratio.
        other parameters (float): Additional parameters as needed.
        initial temperature (float): Initial temperature (K).
        Tint (float): Intake temperature (K).
        Texh (float): Exhaust temperature (K).

    Returns:
    tuple: (mt1, mt2)
        mt1 (numpy array): Array of mass terms corresponding to the crank angles due to intake and exhaust temperatures.
        mt2 (numpy array): Array of mass terms corresponding to the crank angles due to mass flow rates.
    """
    
    #-------------------------------------------------------------------------#
    # Unpacking parameters
    #-------------------------------------------------------------------------#
    case = pars[0]  # Engine case: 'motored' or 'fired'
    n = pars[5] # Engine speed in rps
    Tint = pars[15] # intake temperature
    Texh = pars[17] # exhaust temperature
    
    #-------------------------------------------------------------------------#
    # Gas properties
    #-------------------------------------------------------------------------#
    
    #-------------------------------------------------------------------------#
    # Properties - air or Unburnt/Burnt (Wiebe Function)
    #-------------------------------------------------------------------------#

    if case == 'motored':

        #---------------------------------------------------------------------#
        # air properties
        #---------------------------------------------------------------------#
        airProperties = np.load('./Files/properties-air.npy')

        #---------------------------------------------------------------------#
        # specific heat - J/kgK
        #---------------------------------------------------------------------#
        cv = airProperties[3] # constant-volume specific heat
        cp = airProperties[4] # constant-pressure specific heat
        k = cp/cv # specific heat ratio

    elif case == 'fired':

        #---------------------------------------------------------------------#
        # Mixture - Unburnt/Burnt
        #---------------------------------------------------------------------#

        #---------------------------------------------------------------------#
        # Unburnt State
        #---------------------------------------------------------------------#
        # Load unburnt state properties from file
        unburntProperties = np.load('./Files/properties-unburnt.npy')
        
        #---------------------------------------------------------------------#
        # Burnt State
        #---------------------------------------------------------------------#
        # Load burnt state properties from file
        burntProperties = np.load('./Files/properties-burnt.npy')

        #---------------------------------------------------------------------#
        # Wiebe function
        #---------------------------------------------------------------------#
        xb, _ = wiebeFunction(Th, pars) # Calculate burnt mass fraction using Wiebe function
        xu = 1-xb # unburnt mass fraction

        #---------------------------------------------------------------------#
        # Specific heat - J/kgK
        #---------------------------------------------------------------------#
        cvu = unburntProperties[3] # unburnt constant-volume specific heat
        cvb = burntProperties[3] # burnt constant-volume specific heat
        cpu = unburntProperties[4] # unburnt constant-pressure specific heat
        cpb = burntProperties[4] # burnt constant-pressure specific heat

        cv = xu*cvu + xb*cvb # mixture constant-volume specific heat
        cp = xu*cpu + xb*cpb # mixture constant-pressure specific heat
        k = cp/cv # specific heat ratio

    #-------------------------------------------------------------------------#
    # Intake and exhaust mass flow models
    #-------------------------------------------------------------------------#
    mdotInt = intakeModel(Th, p, T, pars)  # Intake mass flow rate - kg/s
    mdotExh = exhaustModel(Th, p, T, pars) # Exhaust mass flow rate - kg/s
    
    #-------------------------------------------------------------------------#
    # Mass solver
    #-------------------------------------------------------------------------#
    
    #-------------------------------------------------------------------------#
    # Calculate mass term in the temperature ODE
    #-------------------------------------------------------------------------#
    mt1 = (mdotInt*(k*Tint))/(m*2*np.pi*n) - (mdotExh*(k*Texh))/(m*2*np.pi*n)
    mt2 = (mdotInt)/(m*2*np.pi*n) - (mdotExh)/(m*2*np.pi*n)

    return mt1, mt2

#-----------------------------------------------------------------------------#

def temperatureODE(Th, T, m, p, V, dV, pars):
    
    """
    Defines the ODE for temperature as a function of crank angle.
    
    Parameters:
    Th (numpy array): Array of crank angles in radians.
    T (numpy array): Array of temperatures at the corresponding crank angles.
    m (numpy array): Array of masses at the corresponding crank angles.
    p (numpy array): Array of pressures at the corresponding crank angles.
    V (numpy array): Array of cylinder volumes at the corresponding crank angles.
    dV (numpy array): Array of derivatives of cylinder volumes at the corresponding crank angles.
    pars (tuple): Tuple of engine parameters (case, B, S, L, rv, other parameters, initial temperature).

    Returns:
    numpy array: Array of rate of change of temperature with respect to crank angle.
    """
    
    #-------------------------------------------------------------------------#
    # Heat term solver
    #-------------------------------------------------------------------------#
    ht = heatTermSolver(Th, m, p, pars)
    
    #-------------------------------------------------------------------------#
    # Work term solver
    #-------------------------------------------------------------------------#
    wt = workTermSolver(Th, V, dV, pars)
    
    #-------------------------------------------------------------------------#
    # Reaction term solver
    #-------------------------------------------------------------------------#
    rt = reactionTermSolver(Th, pars)
    
    #-------------------------------------------------------------------------#
    # Mass term solver
    #-------------------------------------------------------------------------#
    mt1, mt2 = massTermSolver(Th, m, p, T, pars)

    #-------------------------------------------------------------------------#
    # Energy balance equation
    #-------------------------------------------------------------------------#
    dTdTh = ht - wt*T - rt + mt1 - mt2*T
    
    return dTdTh

#-----------------------------------------------------------------------------#

def idealGasEquationState(Th, V, T, m, pars):
    
    """
    Calculates the pressure profile over crank angles.
    
    Parameters:
    Th (numpy array): Array of crank angles in radians.
    V (numpy array): Array of volume values at different crank angles.
    T (numpy array): Array of temperature values at different crank angles.
    m (numpy array): Array of mass values at different crank angles.
    pars (tuple): Tuple of engine parameters (case, stroke, bore, compression ratio, initial density, initial temperature).
    
    Returns:
    numpy array: Array of pressure values corresponding to the crank angles.
    """
    
    #-------------------------------------------------------------------------#
    # Unpacking parameters
    #-------------------------------------------------------------------------#
    case = pars[0]  # Type of simulation case (e.g., 'motored' or 'fired')
    
    #-------------------------------------------------------------------------#
    # Gas properties
    #-------------------------------------------------------------------------#
    
    #-------------------------------------------------------------------------#
    # Properties - air or Unburnt/Burnt (Wiebe Function)
    #-------------------------------------------------------------------------#

    if case == 'motored':

        #---------------------------------------------------------------------#
        # air properties
        #---------------------------------------------------------------------#
        airProperties = np.load('./Files/properties-air.npy')

        #---------------------------------------------------------------------#
        # Molar mass - kg/kmol
        #---------------------------------------------------------------------#
        M = airProperties[6]
        
    elif case == 'fired':

        #---------------------------------------------------------------------#
        # Mixture - Unburnt/Burnt
        #---------------------------------------------------------------------#

        #---------------------------------------------------------------------#
        # Unburnt State
        #---------------------------------------------------------------------#
        # Load unburnt state properties from file
        unburntProperties = np.load('./Files/properties-unburnt.npy')
        
        #---------------------------------------------------------------------#
        # Burnt State
        #---------------------------------------------------------------------#
        # Load burnt state properties from file
        burntProperties = np.load('./Files/properties-burnt.npy')

        #---------------------------------------------------------------------#
        # Wiebe function
        #---------------------------------------------------------------------#
        xb, _ = wiebeFunction(Th, pars) # Calculate burnt mass fraction using Wiebe function
        xu = 1-xb # unburnt mass fraction

        #---------------------------------------------------------------------#
        # Molar mass - kg/kmol
        #---------------------------------------------------------------------#
        Mu = unburntProperties[6] # unburnt molar mass
        Mb = burntProperties[6] # burnt molar mass
        M = 1/(xu/Mu + xb/Mb) # molar mass of the mixture
        
    #-------------------------------------------------------------------------#
    # The ideal gas constant - J/kmol*K
    #-------------------------------------------------------------------------#
    RU = ct.gas_constant
    
    #-------------------------------------------------------------------------#
    # Gas constant - J/kg*K
    #-------------------------------------------------------------------------#
    R = RU/M
    
    #-------------------------------------------------------------------------#
    # The ideal gas equation of state
    #-------------------------------------------------------------------------#
    p = (m*R*T)/V
    
    return p

#-----------------------------------------------------------------------------#

def systemODE(Th, y, pars):

    """
    Calculates the derivatives of mass and temperature with respect to an independent variable Th
    in a system of ordinary differential equations (ODEs).

    Args:
    Th (float): The independent variable (crank angle).
    y (list or array): A vector containing the state variables [m, T], where
        m (float): The mass.
        T (float): The temperature.
    pars (tuple): A tuple containing system parameters (case, B, S, L, rv, other parameters).

    Returns:
    list: A list containing the derivatives [dmdTh, dTdTh], where
        dmdTh (float): The derivative of mass with respect to Th.
        dTdTh (float): The derivative of temperature with respect to Th.
    """

    #-------------------------------------------------------------------------#
    # state variables: mass (m) and temperature (T) from the vector y
    #-------------------------------------------------------------------------#
    m, T = y
    
    #-------------------------------------------------------------------------#
    # volume (V) and its derivative (dV) with respect to Th
    #-------------------------------------------------------------------------#
    V, dV = volume(Th, pars)
    
    #-------------------------------------------------------------------------#
    # pressure (p) using the ideal gas equation of state
    #-------------------------------------------------------------------------#
    p = idealGasEquationState(Th, V, T, m, pars)
    
    #-------------------------------------------------------------------------#
    # mass ODE
    #-------------------------------------------------------------------------#
    dmdTh = massODE(Th, m, p, T, pars)
    
    #-------------------------------------------------------------------------#
    # temperature ODE
    #-------------------------------------------------------------------------#
    dTdTh = temperatureODE(Th, T, m, p, V, dV, pars)
    
    return [dmdTh, dTdTh]

#-----------------------------------------------------------------------------#

def airState(pars):
    
    """
    Calculates and saves the non-reactive state properties of the air entering the engine, modeled as a mixture of ideal gases.
    
    Parameters:
    pars (tuple): Tuple of engine parameters, where:
        - p0 (float): Pressure in Pascals (Pa).
        - T0 (float): Temperature in Kelvin (K).
    
    Returns:
    air (Cantera Solution): Cantera Solution object representing the air with the calculated properties.
    """
    
    #-------------------------------------------------------------------------#
    # Unpacking parameters
    #-------------------------------------------------------------------------#
    p0 = pars[14] # pressure - Pa
    T0 = pars[15] # temperature - K

    #-------------------------------------------------------------------------#
    # non-reactive state
    #-------------------------------------------------------------------------#
    
    #-------------------------------------------------------------------------#
    # Usando a função 'Solution' do cantera  com o arquivo Air
    #-------------------------------------------------------------------------#
    air = ct.Solution('Air.yaml')
    
    #-------------------------------------------------------------------------#
    # applying temperature and pressure
    air.TP = T0, p0
    #-------------------------------------------------------------------------#

    #-------------------------------------------------------------------------#    
    # Unburnt properties
    #-------------------------------------------------------------------------#
    T, p, X = air.TPX
    hfRT = air.standard_enthalpies_RT # Array of nondimensional species standard-state enthalpies at the current temperature and pressure.
    RU = ct.gas_constant # The ideal gas constant - J/kmol*K
    M = air.mean_molecular_weight # mixture molar mass /mean molecular weight/ - kg/kmol
    R = RU/M # gas constant - J/kgK
    hf = np.sum((hfRT*(R*air.T))*air.X) # Species standard-state enthalpy at the current temperature and pressure.
    uf = (hf-RU*air.T)/M # Species standard-state internal energy at the current temperature and pressure.
    cv = air.cv_mass # constant-volume specific heat - J/kg*K
    cp = air.cp_mass # constant-pressure specific heat - J/kg*K
    k = cp/cv # specific heat ratio

    #-------------------------------------------------------------------------#
    # Array with unburt properties
    #-------------------------------------------------------------------------#
    airProperties = np.array([p, T, k, cv, cp, R, M, uf])
    np.save('./Files/properties-air',(airProperties), allow_pickle = True)
    
    return air

#-----------------------------------------------------------------------------#

def unburntState(pars):
    
    """
    Calculates and saves the properties of the unburnt state in the engine, where fuel is mixed with air to form an ideal gas mixture.
    
    Parameters:
    pars (tuple): Tuple of engine parameters, where:
        - pu (float): Pressure in Pascals (Pa).
        - Tu (float): Temperature in Kelvin (K).
        - phi (float): Equivalence ratio.
        - fuel (str): Type of fuel used.
    
    Returns:
    unburnt (Cantera Solution): Cantera Solution object representing the unburnt mixture with the calculated properties.
    """
    
    #-------------------------------------------------------------------------#
    # Unpacking parameters
    #-------------------------------------------------------------------------#
    pu = pars[14] # pressure - Pa
    Tu = pars[15] # temperature - K
    phi = pars[18] # equivalence ratio
    fuel = pars[19]

    #-------------------------------------------------------------------------#
    # Urburnt State
    #-------------------------------------------------------------------------#
    
    #-------------------------------------------------------------------------#
    # species of GRI 3.0 mechanism
    #-------------------------------------------------------------------------#
    speciesGRI3 = {S.name: S for S in ct.Species.list_from_file('gri30.yaml')}
    
    #-------------------------------------------------------------------------#
    # Create an Ideal Gas object with species that represents combustion
    #-------------------------------------------------------------------------#
    species = [speciesGRI3[S] for S in (str(fuel),'O2','N2')]
    unburnt = ct.Solution(thermo = 'ideal-gas', species = species)
    
    #-------------------------------------------------------------------------#
    # applying the temperature, pressure, and equivalence ratio
    #-------------------------------------------------------------------------#
    unburnt.TP = Tu, pu
    unburnt.set_equivalence_ratio(phi, str(fuel), 'O2:1,N2:3.76')
    
    #-------------------------------------------------------------------------#
    # Unburnt properties
    #-------------------------------------------------------------------------#
    T, p, X = unburnt.TPX
    rho = unburnt.density
    hfRT = unburnt.standard_enthalpies_RT # Array of nondimensional species standard-state enthalpies at the current temperature and pressure.
    RU = ct.gas_constant # The ideal gas constant - J/kmol*K
    M = unburnt.mean_molecular_weight # mixture molar mass /mean molecular weight/ - kg/kmol
    R = RU/M # gas constant - J/kgK
    hf = np.sum((hfRT*(R*unburnt.T))*unburnt.X) # Species standard-state enthalpy at the current temperature and pressure.
    uf = (hf-RU*unburnt.T)/M # Species standard-state internal energy at the current temperature and pressure.
    cv = unburnt.cv_mass # constant-volume specific heat - J/kg*K
    cp = unburnt.cp_mass # constant-pressure specific heat - J/kg*K
    u = unburnt.int_energy_mass # internal energy - J/kg
    h = unburnt.enthalpy_mass # enthalpy - J/kg
    #su = unburnt.entropy_mass
    
    #-------------------------------------------------------------------------#
    # Array with unburt properties
    #-------------------------------------------------------------------------#
    unburntProperties = np.array([p, T, rho, cv, cp, R, M, uf, u, hf, h])
    np.save('./Files/properties-unburnt',(unburntProperties), allow_pickle = True)
    
    return unburnt

#-----------------------------------------------------------------------------#

def burntState(pars):
    
    """
    Calculates and saves the properties of the burnt state in the engine, which is represented by the combustion products forming an ideal gas mixture.
    
    Parameters:
    pars (tuple): Tuple of engine parameters, where:
        - pu (float): Pressure in Pascals (Pa).
        - Tu (float): Temperature in Kelvin (K).
        - phi (float): Equivalence ratio.
        - fuel (str): Type of fuel used.
    
    Returns:
    burnt (Cantera Solution): Cantera Solution object representing the burnt mixture with the calculated properties.
    """
    
    #-------------------------------------------------------------------------#
    # Unpacking parameters
    #-------------------------------------------------------------------------#
    pu = pars[14] # pressure - Pa
    Tu = pars[15] # temperature - K
    phi = pars[18] # equivalence ratio
    fuel = pars[19]

    #-------------------------------------------------------------------------#
    # Burnt State
    #-------------------------------------------------------------------------#
    
    #-------------------------------------------------------------------------#
    # species of GRI 3.0 mechanism
    #-------------------------------------------------------------------------#
    speciesGRI3 = {S.name: S for S in ct.Species.list_from_file('gri30.yaml')}
    
    #-------------------------------------------------------------------------#
    # Create an Ideal Gas object with species that represents combustion
    #-------------------------------------------------------------------------#
    species = [speciesGRI3[S] for S in (str(fuel),'O2','N2','CO2','H2O')]
    burnt = ct.Solution(thermo = 'ideal-gas', species = species)
    
    #-------------------------------------------------------------------------#
    # applying the temperature, pressure, and equivalence ratio
    #-------------------------------------------------------------------------#
    burnt.TP = Tu, pu
    burnt.set_equivalence_ratio(phi, str(fuel), 'O2:1,N2:3.76')
    
    #-------------------------------------------------------------------------#
    # applying the first law for constant enthalpy and pressure
    #-------------------------------------------------------------------------#
    burnt.equilibrate('UV')
    
    #-------------------------------------------------------------------------#
    # Unburnt properties
    #-------------------------------------------------------------------------#
    T, p, X = burnt.TPX
    rho = burnt.density
    hfRT = burnt.standard_enthalpies_RT # Array of nondimensional species standard-state enthalpies at the current temperature and pressure.
    RU = ct.gas_constant # The ideal gas constant - J/kmol*K
    M = burnt.mean_molecular_weight # mixture molar mass /mean molecular weight/ - kg/kmol
    R = RU/M # gas constant - J/kgK
    hf = np.sum((hfRT*(R*burnt.T))*burnt.X) # Species standard-state enthalpy at the current temperature and pressure.
    uf = (hf-RU*burnt.T)/M # Species standard-state internal energy at the current temperature and pressure.
    cv = burnt.cv_mass # constant-volume specific heat - J/kg*K
    cp = burnt.cp_mass # constant-pressure specific heat - J/kg*K
    u = burnt.int_energy_mass # internal energy - J/kg
    h = burnt.enthalpy_mass # enthalpy - J/kg
    #s = burnt.entropy_mass
    
    #-------------------------------------------------------------------------#
    # Array with unburt properties
    #-------------------------------------------------------------------------#
    burntProperties = np.array([p, T, rho, cv, cp, R, M, uf, u, hf, h])
    np.save('./Files/properties-burnt',(burntProperties), allow_pickle = True)
    
    return burnt

#-----------------------------------------------------------------------------#

def ottoCycle(Th, pars):

    """
    Simulates the Otto cycle process given specific parameters.

    Parameters:
    Th : array
        The array of crank angles or times at which the solution is evaluated.
    pars : array
        A list of parameters that includes:
            - case: 'motored' or 'fired'
            - p0: Initial pressure (Pa)
            - T0: Initial temperature (K)
            - Other necessary parameters for the simulation.

    Returns:
    oc : array
        A 2D array containing the volume (V), mass (m), temperature (T), and pressure (p)
        at each crank angle or time.
    """

    #-------------------------------------------------------------------------#
    # Unpacking parameters
    #-------------------------------------------------------------------------#
    case = pars[0]  # Engine case: 'motored' or 'fired'
    p0 = pars[14] # pressure - Pa
    T0 = pars[15] # temperature - K

    #-------------------------------------------------------------------------#
    # definindo os estados termodinâmicos
    #-------------------------------------------------------------------------#
    airState(pars)
    unburntState(pars)
    burntState(pars)

    #-------------------------------------------------------------------------#
    # volume - m³
    #-------------------------------------------------------------------------#
    V, _ = volume(Th, pars)

    #-------------------------------------------------------------------------#
    # initials conditions
    #-------------------------------------------------------------------------#
    if case == 'motored':

        #---------------------------------------------------------------------#
        # air properties
        #---------------------------------------------------------------------#
        R = np.load('./Files/properties-air.npy', allow_pickle=True)[5]

    elif case == 'fired':

        #---------------------------------------------------------------------#
        # unburnt properties
        #---------------------------------------------------------------------#
        R = np.load('./Files/properties-unburnt.npy', allow_pickle=True)[5]

    #-------------------------------------------------------------------------#
    # Solving the ODE system using odeint
    #-------------------------------------------------------------------------#
    V0 = V[0] # volume - m³
    m0 = (p0*V0)/(R*T0) # mass - kg
    sol0 = [m0, T0] # m0 e T0

    #-------------------------------------------------------------------------#

    print('')
    print('Convergence of the %s case'%case)
    for i in tqdm(range(10)):
        sol = odeint(systemODE, y0 = sol0, t = Th, tfirst = True, args = (pars,))
        m0 = sol[:, 0][-1]
        T0 = sol[:, 1][-1]
        sol0 = [m0, T0] # m0 e T0
    
    #-------------------------------------------------------------------------#
    # Obtaining m and T from the solution
    #-------------------------------------------------------------------------#
    m = sol[:, 0]
    T = sol[:, 1]

    #-------------------------------------------------------------------------#
    # pressure - Pa
    #-------------------------------------------------------------------------#
    p = idealGasEquationState(Th, V, T, m, pars)

    #-------------------------------------------------------------------------#
    # solution
    #-------------------------------------------------------------------------#    
    oc = np.array([V, m, T, p])
    
    return oc

#-----------------------------------------------------------------------------#
#=============================================================================#
#-----------------------------------------------------------------------------#
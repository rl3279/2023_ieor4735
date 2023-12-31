import numpy as np
import matplotlib.pyplot as plt

def bond_price_vasicek(
    a:float,
    b:float,
    sigma:float,
    T:int=1,
    r0:float=0.02,
    **kwargs
):  
    """Computes Vasicek bond price using Ricatti ODE"""
    # use A() and B() in Ricatti ODEs
    B = (1 - np.exp(-a*T)) / a
    A = (b/a - (sigma*sigma)/(2*a*a)) * (B - T) - (sigma * sigma * B * B) / (4 * a)
    return np.exp(A - B * r0)

def bond_price(
    r:np.ndarray,
    t:float=0.,
    maturity:float=1.,
    dt:float=1e-2,
    **kwargs
):
    """computes bond price vector using short rate samples"""
    debug = False
    if r.shape[1] < int(maturity // dt)+1:
        raise RuntimeError(
            ("Incorrect input shape. Number of time points "
             "must be sufficient for passsed time increment size."
             f"Minimum: {int(maturity//dt)+1}. Passed: {r.shape[1]}.")
        )
    r_horizon = r[:, round(t / dt):round(maturity/dt)+1]
    if debug:
        print("Orignal r shape:", r.shape)
        print("Passed time period:", f"t={t}, maturity={maturity}")
        print("Result time horizon shape:", r_horizon.shape)
    return np.exp(-(r_horizon * dt).sum(axis = 1))

def forward_term_rate(
    r:np.ndarray,
    t:float=0,
    accr_start:float=3.,
    accr_length:float=0.25,
    dt:float=1e-4,
    **kwargs
):
    """computes forward term rate vector using short rate samples"""
    b1 = bond_price(r, t, accr_start, dt=dt, **kwargs) 
    b2 = bond_price(r, t, accr_start+accr_length, dt=dt, **kwargs)
    return (b1 - b2) / (accr_length * b2)

def forward_term_rate_ratio(
    r:np.ndarray,
    t:float=0.,
    accr_start:float=3.,
    accr_length:float=0.25,
    dt:float=1e-4,
    **kwargs
):
    """This is the L(T,T,T+D) / L(0,T,T+D) ratio in the T-claim.
    The expression is simplified for optimized computation.
    """
    b1 = bond_price(r, accr_start, accr_start+accr_length, dt=dt, **kwargs) 
    b2 = bond_price(r, t, accr_start, dt=dt, **kwargs)
    b3 = bond_price(r, t, accr_start+accr_length, dt=dt, **kwargs)
    return (1 - b1) / (b2 - b3) * b3 / b1

def sim_vasicek(
    a:float,
    b:float,
    sigma:float,
    n:int=1000,
    T:int=1,
    dt:float=1e-2,
    r0:float=0.02,
    **kwargs
):
    """Simulate Vasicek short rate using Gaussian.
    
    Model: dr = (b-ar)dt + sigma*dW
    """
    m = int(T / dt)
    sqrt_dt = np.sqrt(dt)
    dr = np.zeros((n, m))
    r = np.zeros((n, m+1))
    r[:,0] = r0
    Z = np.random.randn(n, m)
    for t in range(m):
        inc = b - a * r[:,t]
        np.copyto(dr[:,t], dt * inc + sigma * Z[:,t] * sqrt_dt)
        np.copyto(r[:,t+1], r[:,t] + dr[:,t])
    return r

def sim_cir(
    a:float,
    b:float,
    sigma:float,
    n:int=1000,
    T:int=1,
    dt:float=1e-2,
    r0:float=0.02,
    **kwargs
):
    """Simulate CIR short rate using Gaussian.

    Model: dr = a(b-r)dt + sig ma*\sqrt{r}dW
    """ 
    m = int(T / dt)
    sqrt_dt = np.sqrt(dt)
    dr = np.zeros((n, m))
    r = np.zeros((n, m+1))
    r[:,0] = r0
    Z = np.random.randn(n, m)
    for t in range(m):
        inc = a * (b - r[:,t])
        np.copyto(dr[:,t], dt * inc + sigma * Z[:,t] * sqrt_dt * np.sqrt(r[:,t]))
        np.copyto(r[:,t+1], r[:,t] + dr[:,t])
    return r

def sim_dothan(
    a: float,
    sigma:float,
    n:int=1000,
    T:int=1,
    dt:float=1e-2,
    r0:float=0.02,
    **kwargs
):
    """Simulate Dothan short rate using Gaussian.

    Model: dr/r = a*dt + sigma*dW
    """
    m = int(T / dt)
    sqrt_dt = np.sqrt(dt)
    dr = np.zeros((n, m))
    r = np.zeros((n, m+1))
    r[:,0] = r0
    Z = np.random.randn(n, m)
    dr = a * dt + sigma * sqrt_dt * Z
    r[:, 1:] = dr.cumsum(axis = 1) + r0
    return r

def sim_holee_1(
    k: float,
    sigma:float,
    n:int=1000,
    T:int=1,
    dt:float=1e-2,
    r0:float=0.02,
    **kwargs
):
    """Simulate Ho-Lee short rate using Gaussian.

    Model: dr = -k*r*dt + sigma * dW
    """
    m = int(T / dt)
    sqrt_dt = np.sqrt(dt)
    dr = np.zeros((n, m))
    r = np.zeros((n, m+1))
    r[:,0] = r0
    Z = np.random.randn(n, m)
    for t in range(m):
        inc = -k * r[:,t]
        np.copyto(dr[:,t], dt * inc + sigma * Z[:,t] * sqrt_dt * np.sqrt(r[:,t]))
        np.copyto(r[:,t+1], r[:,t] + dr[:,t])
    return r

def sim_holee_2(
    k: float,
    sigma:float,
    W_ma:int=10,
    n:int=1000,
    T:int=1,
    dt:float=1e-2,
    r0:float=0.02,
    **kwargs
):
    """Simulate Ho-Lee short rate using Gaussian.

    Model: dr = -k*r*dt + sigma * dW
    """
    m = int(T / dt)
    sqrt_dt = np.sqrt(dt)
    dr = np.zeros((n, m))
    r = np.zeros((n, m+1))
    r[:,0] = r0
    Z = np.random.randn(n, m)
    for t in range(m):
        inc = -k * r[:,t] if t < W_ma else -k * r[:,t-W_ma:t].mean(axis=1)
        np.copyto(dr[:,t], dt * inc + sigma * Z[:,t] * sqrt_dt * np.sqrt(r[:,t]))
        np.copyto(r[:,t+1], r[:,t] + dr[:,t])
    return r

def sim_hw_v_ma(
    a:float,
    sigma:float,
    W_ma:int=30,
    n:int=1000,
    T:int=1,
    dt:float=1e-2,
    r0:float=0.02,
    **kwargs
):
    """Simulate Vasicek short rate using Gaussian.
    
    Model: dr = (b-ar)dt + sigma*dW
    """
    m = int(T / dt)
    sqrt_dt = np.sqrt(dt)
    dr = np.zeros((n, m))
    r = np.zeros((n, m+1))
    r[:,0] = r0
    Z = np.random.randn(n, m)
    for t in range(m):
        b = r[:, max(0,t-W_ma):t].mean(axis=1)
        inc = b - a * r[:,t]
        np.copyto(dr[:,t], dt * inc + sigma * Z[:,t] * sqrt_dt)
        np.copyto(r[:,t+1], r[:,t] + dr[:,t])
    return r




def sim_short_rate(
    kind = "vasicek",
    **kwargs
):
    if kind == "vasicek":
        return sim_vasicek(**kwargs)
    elif kind == "cir":
        return sim_cir(**kwargs)
    elif kind == "dothan":
        return sim_dothan(**kwargs)
    elif kind == "holee1":
        return sim_holee_1(**kwargs)
    elif kind == "holee2":
        return sim_holee_2(**kwargs)
    elif kind == "hw_v_ma":
        return sim_hw_v_ma(**kwargs)
    else:
        return NotImplementedError
    
def eval_quanto(r_d, sigma_s, dt=1/252):
    # Extract short rate sample shape
    num_samples, num_points = r_d.shape
    T = (num_points-1) * dt

    # Time points for the data series
    time_points = np.linspace(0, T, num_points)

    # Numerical integration of r_d(s) over [0, T]
    integral_r_d = np.trapz(r_d, time_points)

    # Second term, which simplifies to sigma_s^2 * T / 2
    integral_sigma_s_squared = 0.5 * sigma_s**2 * T

    # Third term, which is sigma_s * W_T
    integral_sigma_s_W_T = sigma_s * np.random.randn(num_samples)

    # Evaluate final value
    quanto_ratio_sample = np.exp(
        integral_r_d - integral_sigma_s_squared + integral_sigma_s_W_T
    )
    return quanto_ratio_sample

if __name__ == "__main__":
    import time
    params = {
        "a": 0.5,
        "b": 0.04,
        "sigma": 0.2,
        "n":100,
        "t":1.5,
        "T":5.,
        "maturity": 2.25,
        "dt":1e-4,
        "r0":0.02
    }
    start = time.time()
    r = sim_short_rate(kind="vasicek", **params)
    L1 = forward_term_rate(
        r=r, t=3., accr_start=3., accr_length=0.25, dt = 1e-4
    )
    L2 = forward_term_rate(
        r=r, t=0., accr_start=3., accr_length=0.25, dt = 1e-4
    )
    print("L1:", L1)
    print("L2:", L2)
    Ratio = forward_term_rate_ratio(
        r=r, accr_start=3., accr_length=0.25, dt = 1e-4
    )
    print("Method 1 result:", (L1/L2).mean())
    print("Method 2 result:", Ratio.mean())
    print("Total runtime:", time.time() - start)

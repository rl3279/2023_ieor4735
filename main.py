import numpy as np
import sim
import calib

def main(
    model="vasicek", 
    N=1, k=1, kprime=1,n=100000, dt=1/252, 
    T=3, Delta=0.25,
    loud=True
):
    # read daily sofr rates data
    sofr = calib.read_sofr(ma=True)
    
    # calibrate model for both t=0, t=T
    # parameter at t=0, using one year to calibrate
    a0, b0, sigma0 = calib.calibrate(df=sofr, model=model, W=252).iloc[-1].values
    # expected parameter at t=T, using five years to calibrate
    # assume parameters' long term behavior converges to historical trend
    aT, bT, sigmaT = calib.calibrate(df=sofr, model=model, W=len(sofr)).iloc[-1].values
    # extract last sofr observation as r0
    r0 = sofr.loc[len(sofr), "r"]

    # simulate [0,T] using params at t=0
    sample_0 = sim.sim_short_rate(kind=model, a=a0, b=b0, sigma=sigma0, n=n,T=T+Delta, dt=dt, r0=r0)
    # extract simulated r(T) average as r(T) for simulating [T,T+Delta]
    ErT = sample_0.mean(axis=0)[int(T/dt)-1]
    # simulate [T,T+Delta] using params at t=T
    sample_T = sim.sim_short_rate(kind=model, a=aT, b=bT, sigma=sigmaT, n=n,T=Delta, dt=dt, r0=ErT)

    # using simulate short rate sample paths, transform to term rates
    L0 = sim.forward_term_rate(
        r=sample_0, t=0., accr_start=T, accr_length=Delta, dt=dt
    )
    LT = sim.forward_term_rate(
        r=sample_T, t=(T-T), accr_start=(T-T), accr_length=Delta, dt=dt
    )
    # obtain term rate ratio
    L_ratio = (LT/L0)

    # calibrate quanto using data from yfinance
    sigma_s = calib.calibrate_quanto()

    # using calibrate quanto volatility, simulate quanto value
    # use only short rate in [0,T]
    r_d = sample_0[:,:int(T/dt+1)]
    quanto_ratio = sim.eval_quanto(r_d=r_d, sigma_s=sigma_s,dt=dt)

    # finally, evaluate T-claim
    # T-claim: N*max[0, (k-quanto_ratio)*(L_ratio-kprime)]

    # stochastic discount factor
    discount = sim.bond_price(r_d, t=0, maturity=T,dt=dt)

    claim = discount * N * np.maximum(
        (k - quanto_ratio) * (L_ratio - kprime), 0
    )
    if loud:
        print("Model:", model)
        print(f"Parameters at t=0:")
        print(f"\ta = {a0}")
        print(f"\tb = {b0}")
        print(f"\tsigma = {sigma0}")
        print(f"Parameters at t=T:")
        print(f"\ta = {aT}")
        print(f"\tb = {bT}")
        print(f"\tsigma = {sigmaT}")
        print("R0:", r0)
        print("E[RT]:", ErT)
        print("L_ratio mean:", L_ratio.mean())
        print("quanto_ratio mean:", quanto_ratio.mean())
        print("Average bond price p(0,T):", discount.mean())
    return claim

if __name__ == "__main__":

    claim = main(
        model="cir", N=1,k=1,kprime=1,n=100000,dt=1/252,
        T=3, Delta=0.25, loud =True
    )

    print("Claim price (std):", f"{claim.mean()} ({claim.std()})")
        

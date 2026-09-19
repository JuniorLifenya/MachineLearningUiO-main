#!/usr/bin/env python3
"""Independent checks for GW_QSense_Audited_Plan.tex.
Run: python3 verify_gw_plan.py
Requires Python 3, SymPy, NumPy, SciPy. Writes no files unless --json PATH is used.
This checks the specified reduced models, not a complete FW derivation or device.
"""
import argparse, json, math
import numpy as np
import sympy as s
from sympy.physics.hydrogen import R_nl
from scipy.constants import hbar, h, k, c, G, physical_constants
from scipy.integrate import quad, solve_ivp
from scipy.linalg import expm
checks = []
results = {}
def check(name, condition):
    if not bool(condition): raise AssertionError(name)
    checks.append(name)
def eqmat(a,b): return s.simplify(a-b)==s.zeros(*a.shape)
I=s.I; rt=s.sqrt; half=s.Rational(1,2)
pauli=[s.Matrix([[0,1],[1,0]]),s.Matrix([[0,-I],[I,0]]),s.diag(1,-1)]
eye=s.eye(2)
s1=[s.kronecker_product(x/2,eye) for x in pauli]
s2=[s.kronecker_product(eye,x/2) for x in pauli]
P=s.Matrix([[1,0,0],[0,1/rt(2),0],[0,1/rt(2),0],[0,0,1]])
S=[s.simplify(P.H*(a+b)*P) for a,b in zip(s1,s2)]
for a in range(3):
    check('triplet projection s1_%d'%a,eqmat(P.H*s1[a]*P,S[a]/2))
    check('triplet projection s2_%d'%a,eqmat(P.H*s2[a]*P,S[a]/2))
dot=sum((s1[a]*s2[a] for a in range(3)),s.zeros(4))
sdotS=sum((s1[a]*(s1[a]+s2[a]) for a in range(3)),s.zeros(4))
check('s1 dot S in triplet is 1',eqmat(P.H*sdotS*P,s.eye(3)))
for a in range(3):
 for b in range(3):
    T=s1[a]*s2[b]+s1[b]*s2[a]-s.Rational(2,3)*dot*int(a==b)
    Q=(S[a]*S[b]+S[b]*S[a])/2-s.Rational(2,3)*s.eye(3)*int(a==b)
    check('joint spin quadrupole %d%d'%(a,b),eqmat(P.H*T*P,Q))
Qp=S[0]**2-S[1]**2; Qc=S[0]*S[1]+S[1]*S[0]
check('double quantum matrix',eqmat(Qp,s.Matrix([[0,0,1],[0,0,0],[1,0,0]])))
check('cross quantum matrix',eqmat(Qc,s.Matrix([[0,0,-I],[0,0,0],[I,0,0]])))
check('0,+1 projected quadrupole is zero',eqmat(Qp.extract([1,0],[1,0]),s.zeros(2)))
check('Q annihilates ms=0',Qp*s.Matrix([0,1,0])==s.zeros(3,1))
check('single quantum rank2 coupling nonzero',(S[0]*S[2]+S[2]*S[0])[0,1]!=0)
# Exact hydrogen integrals in Bohr-radius units, Condon-Shortley phases.
r,th,ph=s.symbols('r th ph',real=True,positive=True)
Y11=-rt(s.Rational(3,8)/s.pi)*s.sin(th)*s.exp(I*ph)
Y1m=rt(s.Rational(3,8)/s.pi)*s.sin(th)*s.exp(-I*ph)
Y00=1/rt(4*s.pi)
Y22=rt(s.Rational(15,32)/s.pi)*s.sin(th)**2*s.exp(2*I*ph)
angular=lambda a,b:s.simplify(s.integrate(s.integrate(s.expand_complex(s.conjugate(a)*s.sin(th)**2*s.cos(2*ph)*b)*s.sin(th),(ph,0,2*s.pi)),(th,0,s.pi)))
rad2=s.integrate(R_nl(2,1,r)**2*r**4,(r,0,s.oo))
radinv=s.integrate(R_nl(2,1,r)**2*r,(r,0,s.oo))
a2=angular(Y11,Y1m)
check('hydrogen 2p r^2=30 a^2',rad2==30)
check('hydrogen 2p 1/r=1/(4a)',radinv==s.Rational(1,4))
check('hydrogen 2p angular=-2/5',a2==-s.Rational(2,5))
check('hydrogen 2p quadrupole=-12 a^2',rad2*a2==-12)
check('hydrogen 1s to 2p forbidden',angular(Y11,Y00)==0)
rad31=s.integrate(R_nl(3,2,r)*R_nl(1,0,r)*r**4,(r,0,s.oo))
q31=s.simplify(rad31*angular(Y22,Y00))
check('hydrogen 1s to 3d2 Q=81/128 a^2',q31==s.Rational(81,128))
# Bar mass, stiffness, tidal overlap, force and normalization.
x,L,rho,A,E=s.symbols('x L rho A E',positive=True,real=True)
u=s.sin(s.pi*x/L)
integ=lambda f:s.integrate(f,(x,-L/2,L/2))
m1=s.simplify(rho*A*integ(u*u)); J=s.simplify(rho*A*integ(x*u))
stiff=s.simplify(E*A*integ(s.diff(u,x)**2))
check('bar modal mass M/2',s.simplify(m1-rho*A*L/2)==0)
check('bar overlap 2ML/pi^2',s.simplify(J-2*rho*A*L**2/s.pi**2)==0)
check('bar omega0^2',s.simplify(stiff/m1-s.pi**2*E/(rho*L**2))==0)
check('bar mode free end derivative',s.diff(u,x).subs(x,L/2)==0)
num=quad(lambda z:z*np.sin(np.pi*z),-.5,.5,epsabs=1e-13)[0]
check('independent bar quadrature',abs(num-2/np.pi**2)<1e-13)
# Bound-particle double-commutator: harmonic oscillator 0 -> 2 counterexample.
a=s.zeros(6)
for n in range(1,6):a[n-1,n]=rt(n)
xx=(a+a.H)/rt(2); pp=(a-a.H)/(I*rt(2))
check('p^2 identity counterexample',s.simplify((pp**2)[2,0]+(xx**2)[2,0])==0)
check('correct double commutator',s.simplify((pp**2)[2,0]-(1-s.Integer(4)/2)*(xx**2)[2,0])==0)
# Exact finite-dimensional double commutator tested away from cutoff.
H=s.diag(*[s.Rational(2*n+1,2) for n in range(6)])
comm=lambda a,b:a*b-b*a
check('double commutator [2,0]',s.simplify(comm(H,comm(H,xx**2))[2,0]-(-2*pp**2+2*xx**2)[2,0])==0)
# A monochromatic, direction/polarization-specific spectral cross section
# is derived in the PDF. Boughn-Rothman uses an isotropic unpolarized average.
coeff=float(81*s.pi**2/2560)
check('BR average coefficient',abs(coeff-0.312280451753)<1e-12)
results['BR_sigma_over_planck_area']=coeff
results['BR_sigma_m2']=coeff*G*hbar/c**3
# Explicit illustrative values, not a claim that a built detector has these values.
M=1800.; f0=150.; w0=2*np.pi*f0; vs=5000.; length=vs/(2*f0)
Q=1e10; temp=.001; gamma=w0/Q; mass=M/2; aa=2*length/np.pi**2
xzp=np.sqrt(hbar/(2*mass*w0)); ezp=np.pi*xzp/length
thermal=1/np.expm1(hbar*w0/(k*temp))
results.update(M_kg=M,f0_Hz=f0,L_m=length,Q=Q,T_K=temp,
 amplitude_ringup_seconds=2/gamma,amplitude_ringup_days=2/gamma/86400,
 linewidth_Hz=f0/Q,q_zpf_m=xzp,epsilon_zpf=ezp,n_thermal=thermal,
 ground_state_temperature_K=h*f0/k,
 single_NVsimplified_g_Hz=1e10*ezp,
 thermal_excitation_rate_per_s=gamma*thermal,
 thermal_strain_ASD_per_sqrtHz=np.sqrt(4*k*temp*gamma/(mass*aa**2*w0**4)),
 resonant_strain_gain=2*Q/np.pi,
 short_signal_t_s=.2,short_signal_strain_gain=2*f0*.2,
 short_signal_epsilon_for_h1e_21=2*f0*.2*1e-21,
 readout_only_h_ASD_illustration=3e-6/(2*Q/np.pi),
 single_NV_fR_Hz_if_epsilon1e_21=1e10*1e-21,
 single_NV_OmegaR_rad_s_if_epsilon1e_21=2*np.pi*1e10*1e-21,
 pi_time_years_if_epsilon1e_21=1/(2*1e10*1e-21)/(365.25*86400),
 Bz_T_for_double_quantum_150Hz=150/(2*28e9),
 compact_diamond_f0_Hz_if_vs17500_L1mm=17500/(2*.001))
# Independent time-domain integration and harmonic analytic solution.
w0t=1.; Gammat=.04; drive=.7; forcing=0.2
z=forcing/(w0t*w0t-drive*drive-1j*Gammat*drive)
times=np.linspace(0,30,601)
y0=[z.real,( -1j*drive*z).real]
sol=solve_ivp(lambda t,y:[y[1],-Gammat*y[1]-y[0]+forcing*np.cos(drive*t)],(0,30),y0,t_eval=times,rtol=2e-11,atol=2e-13)
truth=np.real(z*np.exp(-1j*drive*times))
err=np.max(np.abs(sol.y[0]-truth));check('bar transfer function vs ODE',err<1e-9)
results['bar_time_domain_max_abs_error']=float(err)
# RK4 order test against exact damped oscillator matrix exponential.
B=np.array([[0.,1.],[-1.,-.1]]); yinit=np.array([1.,0.]); final=5.
def rk4(step):
 y=yinit.copy(); n=round(final/step)
 for _ in range(n):
  k1=B@y;k2=B@(y+step*k1/2);k3=B@(y+step*k2/2);k4=B@(y+step*k3)
  y+=step*(k1+2*k2+2*k3+k4)/6
 return y
truth=expm(B*final)@yinit
errs=[np.linalg.norm(rk4(dt)-truth) for dt in [.1,.05,.025]]
ratios=np.array(errs[:-1])/errs[1:]
check('RK4 fourth-order convergence',np.all((ratios>15.5)&(ratios<16.5)))
results['RK4_error_ratios']=ratios.tolist()
# QFI noncommuting counterexample: G-time-squared shortcut fails for H0 != 0.
sx=np.array([[0,1],[1,0]],complex); sz=np.diag([1,-1]); initial=np.array([1,1],complex)/np.sqrt(2)
t=1.7; theta=.4; eps=1e-5
state=lambda p:expm(-1j*t*(.8*sx+p*sz))@initial
psi=state(theta); dp=(state(theta+eps)-state(theta-eps))/(2*eps)
fq=4*(np.vdot(dp,dp).real-abs(np.vdot(psi,dp))**2)
check('QFI nonnegative',fq>=0)
check('noncommuting QFI not 4t^2 Var(G)',abs(fq-4*t*t)>1.)
# Commuting benchmark H(theta)=theta sigma_z/2, FQ=t^2.
state2=lambda p:expm(-1j*t*p*sz/2)@initial
p2=state2(theta);d2=(state2(theta+eps)-state2(theta-eps))/(2*eps)
fq2=4*(np.vdot(d2,d2).real-abs(np.vdot(p2,d2))**2)
check('commuting QFI=t^2',abs(fq2-t*t)<1e-8)
results['QFI_commuting']=float(fq2);results['QFI_noncommuting']=float(fq)
results['checks_passed']=len(checks);results['checks']=checks
print(json.dumps(results,indent=2))
parser=argparse.ArgumentParser();parser.add_argument('--json');args=parser.parse_args()
if args.json:
 from pathlib import Path
 Path(args.json).write_text(json.dumps(results,indent=2)+'\n')

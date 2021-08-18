from Picca_Pk2xi2D import extrapolate_pk_logspace,HankelTransform,Pk2XiR
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from nbodykit.lab import *
import numpy as np

#initialize the linear power spectrum with the input cosmology
c = cosmology.cosmology.Cosmology(h=0.678,T0_cmb=2.726,Omega0_b=0.0484,Omega0_cdm=0.2596)
c2 = c.from_file('data/lcdm_parameters.ini')
Plin = cosmology.LinearPower(c2, redshift=2.5, transfer='CLASS')

        
def get_sideband_xi(r,xi, fit_range=[[50., 80.], [160., 190.]], 
                        plotit=False):
    ''' Gets correlation function without BAO peak using 
        scipy.optimize.minimize function 
        
        This fucntions was taken from Julian Bautista eboss_clustering repository
    '''
 
    peak_range = [fit_range[0][1], fit_range[1][0]]
    w = ((r>fit_range[0][0])&(r<fit_range[0][1])) | \
        ((r>fit_range[1][0])&(r<fit_range[1][1]))
    x_fit = r[w]
    y_fit = xi[w]

    def broadband(x, *pars):
        xx = x/100
        return pars[0]*xx + pars[1] + pars[2]/xx + pars[3]/xx**2 + \
                pars[4]*xx**2 + pars[5]*xx**3 + pars[6]*xx**4  

    popt, pcov = curve_fit(broadband, x_fit, y_fit,
                            p0=[0, 0, 0, 0, 0, 0, 0])
    
    xi_sideband = xi*1.
    w_peak = (r>peak_range[0])&(r<peak_range[1])
    xi_sideband[w_peak] = broadband(r[w_peak], *popt)

    return xi_sideband

def Pk2D(pars,kp,kt,k,pk_type):
    #get the 2D kaiser power spectrum

    beta,log_b,sigma_t= pars
    b = 10**log_b
    norm = 1/(2*np.pi)**1.5    

    r,xi = HankelTransform(k,Plin(k),q=1.5,mu=0.5,output_r_power=-3,output_r=None,r0=1.)
    xi *= norm
    xi_sb = get_sideband_xi(r,xi)

    # get the sideband and bao pk
    junk, pk_sb = HankelTransform(r,xi_sb,q=1.5,mu=0.5,output_r=k,output_r_power=-3,r0=1.0)
    pk_sb /= norm
    
    kk2 = kp**2 + kt**2
    muk = kp/np.sqrt(kk2)
    pk2d = extrapolate_pk_logspace(np.sqrt(kk2.ravel()),k,Plin(k)).reshape(kk2.shape)
    pk2d_sb = extrapolate_pk_logspace(np.sqrt(kk2.ravel()),k,pk_sb).reshape(kk2.shape)
    pk2d_bao = pk2d - pk2d_sb
 
    A = (b**2) * (1 + beta*muk**2)**2 #this also should be 2d 
    f = c2.Om(2.5)**0.55
    sigma_p = sigma_t * (1.+f)
    #sigma = (mu_p * sigma_p)**2 + (1.0 - mu_p**2) * sigma_t**2
    exponent = np.exp(-(kp**2*sigma_p**2 + kt**2*sigma_t**2) / 2.)
    pk2d_bao *= exponent

    if pk_type == 'bao':
        return A*pk2d_bao
    elif pk_type =='sideband':
        return A*pk2d_sb
    elif pk_type =='total':
        return A*(pk2d_bao + pk2d_sb)

def xi_fast_fixedmu_v2(r_l,r_t,xi_dump,Nmu=4,round_point=3):
    #calculate the wedges from xi(r_par,r_perp)
    #By default the function outputs are 4 wedges
    #To get 8 wedges change Nmu=8, round_point=4

    wedges = np.arange(0,1,1./float(Nmu)) + (1./float(Nmu*2))
    r_tot = np.sqrt(r_l[:,None]**2 + r_t**2)
    mu_val = r_l[:,None]/r_tot
    r_tot = r_tot.flatten()
    mu_val = mu_val.flatten()
    
    mu_val = np.round(mu_val,round_point)
    
    xi_complete = []
    for mu in wedges:
        mask = mu_val == mu
        r_mask = r_tot[mask]
        xi_mask = xi_dump.ravel()[mask]
        mask2 = (r_mask>20.0) & (r_mask<250.0) 
        index = np.argsort(r_mask[mask2])
        r_sort = np.sort(r_mask[mask2])
        xi_mask2 = xi_mask[mask2]
        xi_sort = xi_mask2[index]

        r_spline = np.arange(50,160,5)+2.5 
        xi_spline = InterpolatedUnivariateSpline(r_sort,xi_sort)
        xi_complete.append(xi_spline(r_spline))#*pow(r_spline,2))
    return np.array(xi_complete)


def model(pars):
    n = 2048

    k1d = np.logspace(-4.,1.,n)
    kp = np.tile(k1d,(n,1)).T
    kt = np.tile(k1d,(n,1))

    rp = np.linspace(1e-2,200,n)
    rt = np.linspace(1e-2,200,n)

    p2d_bao = Pk2D(pars[:3],kp,kt,k1d,'bao')
    p2d_sb = Pk2D(pars[:3],kp,kt,k1d,'sideband')

    #Pk2XiR is the picca function to transfrom Pk2D to xi2D
    # note that for the peak term we are multiplying by the alphas    
    xi2d_bao = Pk2XiR(k1d,p2d_bao,pars[-2]*rp,pars[-1]*rt)
    xi2d_sb = Pk2XiR(k1d,p2d_sb,rp,rt)

    xi_bao = xi_fast_fixedmu_v2(rp,rt,np.array(xi2d_bao))#,Nmu=8,round_point=4)
    xi_sb = xi_fast_fixedmu_v2(rp,rt,np.array(xi2d_sb))#,Nmu=8,round_point=4)

    xi_mu = xi_bao + xi_sb
    xi_mu = xi_mu.flatten()
   
    return xi_mu

if __name__ == '__main__':
    pars = [1.2,-0.7,3.29,1.,1.]

    xi = model(pars)
    print(np.shape(xi))
    np.savetxt('../multipoles/wedges_data.dat',xi)
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--outs',type=str) #output dir
parser.add_argument('--file',type=str,help='correlation stacks') #load correlations stack
parser.add_argument('--cov',type=str,help='inverse covariance file') #load precision matrix
parser.add_argument('--type',type=str,default='nl',help='type to run: nl-nonlinear, l-linar, sigmas')
args = parser.parse_args()

data = np.loadtxt(args.file)
mean_data = np.mean(data,axis=0)

icov = np.loadtxt(args.cov)

from model import model

def lnhood(pars):
    coefs = mean_data - model(pars)
    P = np.dot(coefs,np.dot(icov,coefs))
    return -0.5*P

def myprior(cube):
    cube[0] = cube[0]*5.0
    cube[1] = cube[1]*(-4.0 + 1) + 1.0
    cube[2] = cube[2]*8.0
    cube[3] = cube[3]*(1.3-0.8) + 0.8
    cube[4] = cube[4]*(1.3-0.8) + 0.8
    return cube

def myprior_linear(cube):
    cube[0] = cube[0]*5.0
    cube[1] = cube[1]*(-4.0 + 1) + 1.0
    cube[2] = cube[2]*(1.3-0.8) + 0.8
    cube[3] = cube[3]*(1.3-0.8) + 0.8
    return cube

def myprior_sigmas(cube):
    cube[0] = cube[0]*5.0
    cube[1] = cube[1]*(-4.0 + 1) + 1.0
    cube[2] = cube[2]*8.0
    cube[3] = cube[3]*8.0
    cube[4] = cube[4]*(1.3-0.8) + 0.8
    cube[5] = cube[5]*(1.3-0.8) + 0.8
    return cube

if __name__ == "__main__":
    from pymultinest.solve import solve
    import json

    livep = 600 #live points
    se = 0.8 #sampling efficiency 

    if args.type == 'nl':
        parameters = ['beta','log_b','sigma_t','ap','at']
    elif args.type == 'l':
        parameters = ['beta','log_b','ap','at']
    elif args.type == 'sigmas':
        parameters = ['beta','log_b','sigma_p','sigma_t','ap','at']
    

    ndim = len(parameters)
    prefix = args.outs

    with open('%sparams.json' % prefix, 'w') as f:
        json.dump(parameters, f, indent=2)

    if args.type == 'nl':
        result = solve(LogLikelihood=lnhood, Prior=myprior, n_dims=ndim,sampling_efficiency=se,
                       outputfiles_basename=prefix, verbose=True,n_live_points=livep)
    elif args.type == 'l':
        result = solve(LogLikelihood=lnhood, Prior=myprior_linear, n_dims=ndim,sampling_efficiency=se,
                       outputfiles_basename=prefix, verbose=True,n_live_points=livep)
    elif args.type == 'sigmas':
        result = solve(LogLikelihood=lnhood, Prior=myprior_sigmas, n_dims=ndim,sampling_efficiency=se,
                       outputfiles_basename=prefix, verbose=True,n_live_points=livep)

    print()
    print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
    print()
    print('parameter values:')
    for name, col in zip(parameters,result['samples'].transpose()):
        print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))
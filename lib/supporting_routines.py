from copy import deepcopy

import pandas as pn
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import scipy.special
import numpy.linalg as LA



import pamtra2

niceKeys = {
    'Nw_log10' : 'log$_{10}$ N$_w$',
    'Dm_log10' : 'log$_{10}$ D$_m$',
    'Sm_log10' : 'log$_{10}$ $\sigma_m$',
    'Nw' : 'N$_w$',
    'Dm' : 'D$_m$',
    'Sm' : '$\sigma_m$',
    'Smprime' : "$\sigma_m\!'$",
    'Sm_prime' : "$\sigma_m\!'$",
    'Smprime_log10' : "log$_{10}$ $\sigma_m\!'$",
    'PCS0' : 'PCS 0',
    'PCS1' : 'PCS 1',
    'PCS2' : 'PCS 2',
}
niceKeysSimple = {
    'Nw' : 'N$_w$',
    'Dm' : 'D$_m$',
    'Sm' : "$\sigma_m$ or $\sigma_m\!'$",
    'Smprime' : "$\sigma_m\!'$",
    'PCS0' : 'PCS 0',
    'PCS1' : 'PCS 1',
    'PCS2' : 'PCS 2',
}
niceRuns = {
    'Sm':'log$_{10}$', 
    'SmLin':"linear", 
    'Smprime':"linear with $\sigma_m\!'$", 
    'SmprimeLog10':"log$_{10}$ with $\sigma_m\!'$",
    'PCS':'PCS',
}
niceRetrievals = {
    'Z': "$Z_e$ retrieval",
    'ZW': "$Z_e$, $V_d$ retrieval",
    'Zdual': "dual $Z_e$ retrieval",
    'ZWdual': "dual $Z_e$, $V_d$ retrieval",

}

def plotCorrelation(cov, fig, sp, tickLabels=None, isCov=True,cmap='viridis_r'):

    std = pn.Series(np.sqrt(np.diag(cov)), index=cov.index)

    if isCov:
        cor = deepcopy(cov)
        cor[:] = 0
        for xx in cov.index:
            for yy in cov.index:
                cor[xx][yy] = cov[xx][yy] / (std[xx] * std[yy])
    else:
        cor = cov

    sp.set_aspect('equal')

    ind_array = np.arange(cor.shape[0])

    x, y = np.meshgrid(ind_array, ind_array)
    for i, (x_val, y_val) in enumerate(zip(x.flatten(), y.flatten())):
        if x_val < y_val:
            c = "%.g" % cov.iloc[x_val, y_val]
            sp.text(
                x_val + 0.5,
                y_val + 0.5,
                c,
                va='center',
                ha='center',
                fontsize=9)
            
        if x_val > y_val:
            c = "%.g" % cor.iloc[x_val, y_val]
            sp.text(
                x_val + 0.5,
                y_val + 0.5,
                c,
                va='center',
                ha='center',
                color='w',
                fontsize=9)
            cor.iloc[x_val, y_val] = -1
            
        if x_val == y_val:
            c = "%.g" % cov.iloc[x_val, y_val]
            sp.text(
                x_val + 0.5,
                y_val + 0.5,
                c,
                va='center',
                ha='center',
                color='k',
                fontsize=9)
            cor.iloc[x_val, y_val] = -1



    if tickLabels != None:
        labels = []
        for ii in range(cor.shape[0]):
            labels.append(tickLabels[cor.index[ii]].split(' [')[0])
    else:
        labels = cov.index
    cor_values = np.ma.masked_equal(cor.values, -1)
    pc = sp.pcolormesh(cor_values, vmin=-1, vmax=1, cmap=cmap)
    sp.tick_params(axis=u'both', which=u'both', length=0)
    sp.set_xticks(np.arange(len(labels)) + 0.5)
    sp.set_xticklabels(labels, rotation=90)
    sp.set_yticks(np.arange(len(labels)) + 0.5)
    sp.set_yticklabels(labels)
    sp.set_xlim(0, len(std))
    sp.set_ylim(0, len(std))
    return pc




def normalizedDSD(D,Nw,Dm,mu):

    
    fmu = (6*(4 + mu)**(mu + 4))/ (4**4 * scipy.special.gamma(mu + 4));
    N = Nw * fmu * ((D/Dm)**mu) * np.exp(-(mu+4) * (D/Dm))
    return N

def normalizedDSD_sigma_prime(D,Nw,Dm,sigma_prime):

    bm = 1.36
    mu = Dm**(2-2*bm)/sigma_prime**2 - 4 # eq. 25 w14
#     print(mu)
    return normalizedDSD(D,Nw,Dm,mu)



def normalizedDSD_sigma(D,Nw,Dm,sigma):

    mu = (Dm/sigma)**2 -4 #eq 18 w14
#     print(mu)
    return normalizedDSD(D,Nw,Dm,mu)

def normalizedDSD4Pamtra(sizeCenter,sizeBoundsWidth,Nw,Dm,mu):
    sizeCenter = sizeCenter*1000.
    
    fmu = (6*(4 + mu)**(mu + 4))/ (4**4 * scipy.special.gamma(mu + 4));
    N = Nw * fmu * ((sizeCenter/Dm)**mu) * np.exp(-(mu+4) * (sizeCenter/Dm))
    
    N = N*1000
    
    return N * sizeBoundsWidth

def preparePamtra(frequencies=[35.9e9],additionalDims={},radar='simple'):
    pam2 = pamtra2.pamtra2(
        nLayer=1,
        hydrometeors=['rain'],
        frequencies=frequencies,
        additionalDims=additionalDims,
    )

    pam2.profile['height'][:] = 100
    pam2.profile['heightBinDepth'] = xr.ones_like(pam2.profile['height']) * 10
    pam2.profile['temperature'][:] = 283
    pam2.profile['pressure'][:] = 100000
    pam2.profile['relativeHumidity'][:] = 50
    pam2.profile['eddyDissipationRate'][:] = 1e-4
    pam2.profile['horizontalWind'][:] = 10

    pam2.addMissingVariables()

    pam2.addHydrometeor(
        pamtra2.hydrometeors.rain(
            name='rain',
            Dmin=1e-5,
            Dmax=1e-2,
            scattering=pamtra2.hydrometeors.scattering.Mie,
            numberConcentration=xr.DataArray([0.] * 50, dims=['sizeBin']),
            useFuncArgDefaults=False,
        ))

    if radar == 'simple':
        pam2.addInstrument(pamtra2.instruments.radar.simpleRadar(name='radar'), solve=False)
    elif radar == 'spectral':
        pam2.addInstrument(pamtra2.instruments.radar.dopplerRadarPamtra(
            name='radar',
            radarMaxV=7.885*2,
            radarMinV=-7.885*2,
            radarNFFT=256*2,
            momentsNPeaks=1,
            seed = 10,
            radarNAve = 15,
        ), solve=False)
    return pam2


def forwardPamtra(X,
                     pam2=None,
                     y_vars=None,
                     solver_training=None,
                     origStd_training=None,
                     origMean_training=None,
                     returnDSD=False,
                 ):
    
    if 'PCS1' in X.keys():
        X = X[['PCS0', 'PCS1', 'PCS2']]
        back2state1 = X.values.dot(LA.inv(solver_training.eofs().T))
        back2state = back2state1 * origStd_training
        back2state = back2state + origMean_training
        X = back2state.to_series()

    try:
        Nw = 10**X['Nw_log10']
    except KeyError:
        Nw = X['Nw']
    try:
        Dm = 10**X['Dm_log10']
    except KeyError:
        Dm = X['Dm']

    if 'Smprime' in X.keys():
        sigma_prime = X['Smprime']
    elif 'Smprime_log10' in X.keys():
        sigma_prime = 10**X['Smprime_log10']
    elif 'Sm_log10' in X.keys():
        sigma = 10**X['Sm_log10']
        sigma_prime = sigma / (Dm**1.36)
    elif 'Sm' in X.keys():
        sigma = X['Sm']
        sigma_prime = sigma / (Dm**1.36)
    else:
        raise KeyError

    bm = 1.36
    mu = Dm**(2 - 2 * bm) / sigma_prime**2 - 4  # eq. 25 w14

    pam2.hydrometeors.rain.profile.numberConcentration[:] = normalizedDSD4Pamtra(
        pam2.hydrometeors.rain.profile.sizeCenter.values,
        pam2.hydrometeors.rain.profile.sizeBoundsWidth.values, Nw, Dm, mu)

    pam2.instruments.radar.solve()

    res = pam2.instruments.radar.results[[
        'radarReflectivity', 'meanDopplerVel'
    ]].load()

    out = pn.Series()
    for freq in pam2.profile.frequency.values:
        out['Ze_%g' % (freq / 1e9)] = res['radarReflectivity'].sel(
            frequency=freq).values.squeeze()
        out['MDV_%g' %
            (freq / 1e9
             )] = res['meanDopplerVel'].sel(frequency=freq).values.squeeze()
    try:
        out = out.astype(np.float64)
    except ValueError:
        pass

    if returnDSD: # for debugging only
        return out[y_vars], (Nw, Dm, sigma_prime)
    else:
        return out[y_vars]
    

def splitTQ(x):
    t_index = [i for i in x.index if i.endswith('t')]
    q_index = [i for i in x.index if i.endswith('q')]
    h_index = [float(i.split('_')[0]) for i in x.index if i.endswith('q')]

    assert len(t_index) == len(q_index)
    assert len(t_index) == len(h_index)
    assert len(t_index)*2 == len(x)


    xt = x[t_index]
    xt.index = h_index

    xq = x[q_index]
    xq.index = h_index
    
    xt.index.name = 'height'
    xq.index.name = 'height'
    
    return xt, xq

def plotMwrResults(oe1, title=None, oe2=None, h=None, hlabel='Height [m]'):
    
    if oe2 is None:
        gridspec = dict(wspace=0.0)        
        fig, (axA,axB) = plt.subplots(ncols=2, sharey=True, gridspec_kw=gridspec, figsize = [5.0, 4.0])
        vals = [oe1], [axA], [axB]
    else:
        
        gridspec = dict(wspace=0.0, width_ratios=[1, 1, 0.25, 1, 1])        
        fig, (axA,axB, ax0, axC, axD) = plt.subplots(ncols=5, sharey=True, figsize = [10.0, 4.0], gridspec_kw=gridspec)
        vals = [oe1, oe2], [axA,axC], [axB, axD]
        ax0.set_visible(False)

        
    for oe, ax1, ax2 in zip(*vals):
        
        t_op, q_op = splitTQ(oe.x_op)
        t_op_err, q_op_err = splitTQ(oe.x_op_err)
        t_a, q_a = splitTQ(oe.x_a)
        t_a_err, q_a_err = splitTQ(oe.x_a_err)
        t_truth, q_truth = splitTQ(oe.x_truth)

        nProf = len(t_op)

        if h is None:
            hvar = t_op.index
        else:
            hvar = h
            
        ax1.plot(t_op, hvar, color='C0', label='Optimal')
        ax1.fill_betweenx(hvar,t_op+t_op_err,t_op-t_op_err,
                        color='C0', alpha=0.2)

        ax1.plot(t_a, hvar, color='C1', label='Prior')
        ax1.fill_betweenx(hvar,t_a+t_a_err,t_a-t_a_err,
                        color='C1', alpha=0.2)
        ax1.plot(t_truth, hvar, color='C2', label='Truth')

        ax2.plot(q_op, hvar, color='C0')
        ax2.fill_betweenx(hvar,q_op+q_op_err,q_op-q_op_err,
                        color='C0', alpha=0.2)

        ax2.plot(q_a, hvar, color='C1')
        ax2.fill_betweenx(hvar,q_a+q_a_err,q_a-q_a_err,
                        color='C1', alpha=0.2)
        ax2.plot(q_truth, hvar, color='C2')


        ax1.set_xlabel('Temperature [K]')
        ax2.set_xlabel('Specific humidity [g/kg]')
    if h is not None:
        axA.invert_yaxis()    

    axA.set_ylabel(hlabel)

    axA.legend(loc='upper right')
    
    fig.suptitle(title)
    return fig


def q2a(q, p, T):
    '''
    specific to absolute humidty
    '''
    Rair = 287.04  # J/kg/K
    Rvapor = 461.5  # J/kg/K
    rho = p / (Rair * T * (1 + (Rvapor / Rair - 1) * q)) #density kg/m3
    return q*rho


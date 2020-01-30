# coding: utf-8
import numpy as np
import numba


'''
Simple microwave radiative transfer code.
Translated from IDL to Python 3 by M. Maahn
'''


def STP_IM10(
        # [m] states grid of T_final [K], p_final [Pa], q_final [kgm^-3]
        z_final,
        T_final,
        p_final,
        q_final,
        theta,  # zenith angle of observation in deg.
        f,  # frequency vector in GHz
        # re-calculate opt. depth for every angle =0: no! (=1: yes is default),
        # tau_calc=True,
        # can save some time when calc. Jacobians
):
    '''
    non-scattering microwave radiative transfer using Rosenkranz 1998 gas
    absorption

    Author: Ulrich Loehnert (loehnert@meteo.uni-koeln.de)

    '''

    z_final = np.asarray(z_final)
    T_final = np.asarray(T_final)
    p_final = np.asarray(p_final)
    q_final = np.asarray(q_final)
    f = np.asarray(f)

    theta = np.deg2rad(theta)
    mu = np.cos(theta) + 0.025 * np.exp(-11. * np.cos(theta))

    # ****radiative transfer

    # ULI: Where is tau, tau_wv, tau_o2 coming from if tau_calc=False? are
    # these global variables?
    # if tau_calc:
    tau, tau_wv, tau_o2 = TAU_CALC_IM10(z_final, T_final, p_final, q_final,
                                        f)

    TB = TB_CALC_PL_IM10(T_final, tau, mu, f)

    return (
        TB,  # [K] brightness temperature array of f grid
        tau,  # total optical depth
        tau_wv,  # WV optical depth
        tau_o2,
    )


@numba.jit(cache=True, nopython=True)
def TAU_CALC_IM10(
    z,        # height [m]
    T,        # Temp. [K]
    p,        # press. [Pa]
    rhow,     # abs. hum. [kg m^-3]
    f,        # freq. [GHz]
):
    '''
    $Id: tau_calc_r98.pro,v 1.1 2009/11/11 14:38:52 loehnert Exp $
    Abstract:
    subroutine to determine optical thichkness tau
    at height k (index counting from bottom of zgrid)
    on the basis of Rosenkranz 1998 water vapor and absorption model
    Rayleigh calculations

    Author: Ulrich Loehnert
    Changes:
    2009-10-09:
    '''

    kmax = len(z)
    n_f = len(f)

    abs_all = np.zeros((kmax-1, n_f))
    abs_wv = np.zeros((kmax-1, n_f))
    abs_o2 = np.zeros((kmax-1, n_f))

    tau = np.zeros((kmax-1, n_f))
    tau_wv = np.zeros((kmax-1, n_f))
    tau_o2 = np.zeros((kmax-1, n_f))

    for ii in range(kmax-1):
        # FOR ii = 0, kmax-2 DO BEGIN
        # alles SI!!
        deltaz = z[kmax-1-ii]-z[kmax-1-ii-1]
        T_mean = (T[kmax-1-ii] + T[kmax-1-ii-1])/2.
        deltap = p[kmax-1-ii]-p[kmax-1-ii-1]

        if deltap >= 0:
            p[kmax-1-ii] = p[kmax-1-ii] - 0.1
            if deltap >= 1:
                print(
                    'Warning: p profile adjusted by %f5.2 to assure monotonic'
                    'decrease!', deltap)

        xp = -np.log(p[kmax-1-ii]/p[kmax-1-ii-1])/deltaz
        p_mean = -p[kmax-1-ii-1]/xp*(np.exp(-xp*deltaz)-1.0)/deltaz
        rhow_mean = (rhow[kmax-1-ii] + rhow[kmax-1-ii-1])/2.

        # ****gas absorption
        # water vapor
        AWV = ABWVR98_IM10(rhow_mean*1000., T_mean, p_mean/100, f)
        AWV = AWV/1000.

        # oxygen
        AO2 = ABO2R98_IM10(T_mean, p_mean/100., rhow_mean*1000., f)
        AO2 = AO2/1000.

        # nitrogen (nur bei Rosenkranz O2)
        AN2 = ABSN2_IM10(T_mean, p_mean/100, f)
        AN2 = AN2/1000.

        absg = AWV + AO2 + AN2

        abs_all[kmax-2-ii, :] = absg
        abs_wv[kmax-2-ii, :] = AWV
        abs_o2[kmax-2-ii, :] = AO2

        tau_x = np.zeros(n_f)
        tau_x1 = np.zeros(n_f)
        tau_x2 = np.zeros(n_f)

        for jj in range(ii+1):

            deltaz = z[kmax-1-jj]-z[kmax-2-jj]
            tau_x = (abs_all[kmax-2-jj, :])*deltaz + tau_x
            tau_x1 = (abs_wv[kmax-2-jj, :])*deltaz + tau_x1
            tau_x2 = (abs_o2[kmax-2-jj, :])*deltaz + tau_x2

        tau[kmax-2-ii, :] = tau_x
        tau_wv[kmax-2-ii, :] = tau_x1
        tau_o2[kmax-2-ii, :] = tau_x2

    return (
        tau,      # total opt. depth
        tau_wv,   # WV opt. depth
        tau_o2    # O2 opt. depth
    )


@numba.jit(cache=True, nopython=True)
def ABWVR98_IM10(
        RHO,  # abs. humidity in gm-3
        T,  # temp. in K
        P,  # pressure in hPa
        F,  # freqeuncy in GHz
):
    '''
    OUPUT:
    ALPHA                        absorption coefficient in nepers(??)/km
    KEYWORDS:
     Abstract:
    PURPOSE- COMPUTE ABSORPTION COEF IN ATMOSPHERE DUE TO WATER VAPOR

    CALLING SEQUENCE PARAMETERS-SPECIFICATIONS

          NAME    UNITS    I/O  DESCRIPTON            VALID RANGE
          T       KELVIN    I   TEMPERATURE
          P       MILLIBAR  I   PRESSURE              .1 TO 1000
          RHO     G/M**3    I   WATER VAPOR DENSITY
          F       GHZ       I   FREQUENCY             0 TO 800
          ALPHA   NEPERS/KM O   ABSORPTION COEFFICIENT

       REFERENCES-
       P.W. ROSENKRANZ, RADIO SCIENCE V.33, PP.919-928 (1998) V.34, P.1025
     (1999).

       LINE INTENSITIES SELECTION THRESHOLD=
         HALF OF CONTINUUM ABSORPTION AT 1000 MB.
       WIDTHS MEASURED AT 22, 183, 380 GHZ, OTHERS CALCULATED.
         A.BAUER ET AL.ASA WORKSHOP (SEPT. 1989) (380GHz).
    #
     Dependencies:
     -
     Changes:
        DATE- OCT.6, 1988  P.W.ROSENKRANZ - EQS AS PUBL. IN 1993.
              OCT.4, 1995  PWR- USE CLOUGH'S DEFINITION OF LOCAL LINE
                       CONTRIBUTION,  HITRAN INTENSITIES, ADD 7 LINES.
              OCT. 24, 95  PWR -ADD 1 LINE.
              JULY 7, 97   PWR -SEPARATE COEFF. FOR SELF-BROADENING,
                           REVISED CONTINUUM.
              DEC. 11, 98  PWR - ADDED COMMENTS
              MAY 30, 2008 Bernhard Pospichal,
              according to Liljegren et al.,2005, IEEE Transactions on 
              Geoscience and Remote Sensing, Vol. 43, No. 5:
              22.235 GHz air-broadened width parameter changed from .00281 to
               .002656.
              Self-broadened width parameter changed from .01349 to .0127488
              11 NOV, 2009 Ulrich Loehnert
              1.) Added multipliers for corrected self (bs_mult) and foreign
               (bf_mult) continuum contribution according to Turner et al. 
               2009 (IEEE TGARS)
              2.) added keywords linew_22 and cont_corr in order to be able to
              choose between original R98 and modified Liljegren 2005 
              linewidth (default), respectively bewteen original and Turner et
              al. modified contiuum (default)
     changed program ...
    -
    '''
    linew_22 = 'lil05'
    cont_corr = 'tur09'

    # ****number of frequencies
    n_f = len(F)

    # ****LOCAL VARIABLES:
    NLINES = 15
    DF = np.zeros((2, n_f))

    # ****LINE FREQUENCIES:
    FL = [
        22.2351, 183.3101, 321.2256, 325.1529, 380.1974, 439.1508, 443.0183,
        448.0011, 470.8890, 474.6891, 488.4911, 556.9360, 620.7008, 752.0332,
        916.1712
    ]

    # ****LINE INTENSITIES AT 300K:
    S1 = [
        .1310E-13, .2273E-11, .8036E-13, .2694E-11, .2438E-10, .2179E-11,
        .4624E-12, .2562E-10, .8369E-12, .3263E-11, .6659E-12, .1531E-08,
        .1707E-10, .1011E-08, .4227E-10
    ]

    # ****T COEFF. OF INTENSITIES:
    B2 = [
        2.144, .668, 6.179, 1.541, 1.048, 3.595, 5.048, 1.405, 3.597, 2.379,
        2.852, .159, 2.391, .396, 1.441
    ]

    # ****AIR-BROADENED WIDTH PARAMETERS AT 300K:
    # W3 = [
    #     .00281, .00281, .0023, .00278, .00287, .0021, .00186, .00263, .00215,
    #     .00236, .0026, .00321, .00244, .00306, .00267
    # ]

    # ****T-EXPONENT OF AIR-BROADENING:
    X = [
        .69, .64, .67, .68, .54, .63, .60, .66, .66, .65, .69, .69, .71, .68,
        .70
    ]

    # ****SELF-BROADENED WIDTH PARAMETERS AT 300K
    # WS = [
    #     .01349, .01491, .0108, .0135, .01541, .0090, .00788, .01275, .00983,
    #     .01095, .01313, .01320, .01140, .01253, .01275
    # ]

    # if linew_22 == 'lil05':
    # ****AIR-BROADENED WIDTH PARAMETERS AT 300K:
    W3 = [
        .002656, .00281, .0023, .00278, .00287, .0021, .00186, .00263,
        .00215, .00236, .0026, .00321, .00244, .00306, .00267
    ]

    # ****SELF-BROADENED WIDTH PARAMETERS AT 300K
    WS = [
        .0127488, .01491, .0108, .0135, .01541, .0090, .00788, .01275,
        .00983, .01095, .01313, .01320, .01140, .01253, .01275
    ]


# ****T-EXPONENT OF SELF-BROADENING:
    XS = [
        .61, .85, .54, .74, .89, .52, .50, .67, .65, .64, .72, 1.0, .68, .84,
        .78
    ]

    if RHO <= 0:
        ALPHA = np.zeros(n_f)
    else:
        PVAP = RHO * T / 217.
        PDA = P - PVAP
        DEN = 3.335E16 * RHO
        TI = 300. / T
        TI2 = TI**2.5

        # ****CONTINUUM TERMS
        bf_org = 5.43E-10
        bs_org = 1.8E-8

        # bf_mult = 1.0
        # bs_mult = 1.0

        # if cont_corr == 'tur09':
        bf_mult = 1.105
        bs_mult = 0.79

        bf = bf_org * bf_mult
        bs = bs_org * bs_mult

        CON = (bf * PDA * TI**3 + bs * PVAP * TI**7.5) * PVAP * F * F

        # ****ADD RESONANCES
        SUM = np.zeros(n_f)

        for I in range(NLINES):

            WIDTH = W3[I] * PDA * TI**X[I] + WS[I] * PVAP * TI**XS[I]
            WSQ = WIDTH * WIDTH
            S = S1[I] * TI2 * np.exp(B2[I] * (1. - TI))
            DF[0, :] = F - FL[I]
            DF[1, :] = F + FL[I]

            # USE CLOUGH'S DEFINITION OF LOCAL LINE CONTRIBUTION
            BASE = WIDTH / (562500. + WSQ)

            # DO FOR POSITIVE AND NEGATIVE RESONANCES
            RES = np.zeros(n_f)

            for i_n_f in range(n_f):
                for J in range(2):
                    if (np.abs(DF[J, i_n_f]) < 750.):
                        RES[i_n_f] = RES[i_n_f] + WIDTH / (
                            DF[J, i_n_f]**2 + WSQ) - BASE

            SUM = SUM + S * RES * (F / FL[I])**2

        ALPHA = .3183E-4 * DEN * SUM + CON

    return ALPHA


@numba.jit(cache=True, nopython=True)
def ABO2R98_IM10(TEMP, PRES, VAPDEN, FREQ):
    '''
    #
    PURPOSE: RETURNS ABSORPTION COEFFICIENT DUE TO OXYGEN IN AIR,
             IN NEPERS/KM
    #
     5/1/95  P. Rosenkranz
     11/5/97  P. Rosenkranz - 1- line modification.
     12/16/98 pwr - updated submm freq's and intensities from HITRAN96
    #
    ARGUMENTS:
    TEMP, PRES, VAPDEN, FREQ
    NAME    UNITS    DESCRIPTION        VALID RANGE
    #
    TEMP    KELVIN   TEMPERATURE        UNCERTAIN, but believed to be
                                         valid for atmosphere
    PRES   MILLIBARS PRESSURE           3 TO 1000
    VAPDEN  G/M^3   WATER VAPOR DENSITY  (ENTERS LINEWIDTH CALCULATION
                     DUE TO GREATER BROADENING EFFICIENCY OF H2O)
    FREQ    GHZ      FREQUENCY          0 TO 900
    #
    REFERENCES FOR EQUATIONS AND COEFFICIENTS:
    P.W. Rosenkranz, CHAP. 2 and appendix, in ATMOSPHERIC REMOTE SENSING
     BY MICROWAVE RADIOMETRY (M.A. Janssen, ed., 1993).
    H.J. Liebe et al, JQSRT V.48, PP.629-643 (1992).
    M.J. Schwartz, Ph.D. thesis, M.I.T. (1997).
    SUBMILLIMETER LINE INTENSITIES FROM HITRAN96.
    This version differs from Liebe's MPM92 in two significant respects:
    1. It uses the modification of the 1- line width temperature dependence
    recommended by Schwartz: (1/T).
    2. It uses the same temperature dependence (X) for submillimeter
    line widths as in the 60 GHz band: (1/T)^0.8

    LINES ARE ARRANGED 1-,1+,3-,3+,ETC. IN SPIN-ROTATION SPECTRUM
    '''
    F = [
        118.7503, 56.2648, 62.4863, 58.4466, 60.3061, 59.5910, 59.1642,
        60.4348, 58.3239, 61.1506, 57.6125, 61.8002, 56.9682, 62.4112, 56.3634,
        62.9980, 55.7838, 63.5685, 55.2214, 64.1278, 54.6712, 64.6789, 54.1300,
        65.2241, 53.5957, 65.7648, 53.0669, 66.3021, 52.5424, 66.8368, 52.0214,
        67.3696, 51.5034, 67.9009, 368.4984, 424.7632, 487.2494, 715.3931,
        773.8397, 834.1458
    ]

    S300 = [
        .2936E-14, .8079E-15, .2480E-14, .2228E-14, .3351E-14, .3292E-14,
        .3721E-14, .3891E-14, .3640E-14, .4005E-14, .3227E-14, .3715E-14,
        .2627E-14, .3156E-14, .1982E-14, .2477E-14, .1391E-14, .1808E-14,
        .9124E-15, .1230E-14, .5603E-15, .7842E-15, .3228E-15, .4689E-15,
        .1748E-15, .2632E-15, .8898E-16, .1389E-15, .4264E-16, .6899E-16,
        .1924E-16, .3229E-16, .8191E-17, .1423E-16, .6494E-15, .7083E-14,
        .3025E-14, .1835E-14, .1158E-13, .3993E-14
    ]

    BE = [
        .009, .015, .083, .084, .212, .212, .391, .391, .626, .626, .915, .915,
        1.260, 1.260, 1.660, 1.665, 2.119, 2.115, 2.624, 2.625, 3.194, 3.194,
        3.814, 3.814, 4.484, 4.484, 5.224, 5.224, 6.004, 6.004, 6.844, 6.844,
        7.744, 7.744, .048, .044, .049, .145, .141, .145
    ]
    # WIDTHS IN MHZ/MB

    WB300 = .56
    X = .8
    W300 = [
        1.63, 1.646, 1.468, 1.449, 1.382, 1.360, 1.319, 1.297, 1.266, 1.248,
        1.221, 1.207, 1.181, 1.171, 1.144, 1.139, 1.110, 1.108, 1.079, 1.078,
        1.05, 1.05, 1.02, 1.02, 1.00, 1.00, .97, .97, .94, .94, .92, .92, .89,
        .89, 1.92, 1.92, 1.92, 1.81, 1.81, 1.81
    ]

    Y300 = [
        -0.0233, 0.2408, -0.3486, 0.5227, -0.5430, 0.5877, -0.3970, 0.3237,
        -0.1348, 0.0311, 0.0725, -0.1663, 0.2832, -0.3629, 0.3970, -0.4599,
        0.4695, -0.5199, 0.5187, -0.5597, 0.5903, -0.6246, 0.6656, -0.6942,
        0.7086, -0.7325, 0.7348, -0.7546, 0.7702, -0.7864, 0.8083, -0.8210,
        0.8439, -0.8529, 0., 0., 0., 0., 0., 0.
    ]

    V = [
        0.0079, -0.0978, 0.0844, -0.1273, 0.0699, -0.0776, 0.2309, -0.2825,
        0.0436, -0.0584, 0.6056, -0.6619, 0.6451, -0.6759, 0.6547, -0.6675,
        0.6135, -0.6139, 0.2952, -0.2895, 0.2654, -0.2590, 0.3750, -0.3680,
        0.5085, -0.5002, 0.6206, -0.6091, 0.6526, -0.6393, 0.6640, -0.6475,
        0.6729, -0.6545, 0., 0., 0., 0., 0., 0.
    ]

    TH = 300. / TEMP
    TH1 = TH - 1.
    B = TH**X
    PRESWV = VAPDEN * TEMP / 217.
    PRESDA = PRES - PRESWV
    DEN = .001 * (PRESDA * B + 1.1 * PRESWV * TH)
    DENS = .001 * (PRESDA + 1.1 * PRESWV) * TH
    DFNR = WB300 * DEN
    SUM = 1.6E-17 * FREQ * FREQ * DFNR / (TH * (FREQ * FREQ + DFNR * DFNR))

    for K in range(40):
        if K == 0:
            DF = W300[0] * DENS
        else:
            DF = W300[K] * DEN
        Y = .001 * PRES * B * (Y300[K] + V[K] * TH1)
        STR = S300[K] * np.exp(-BE[K] * TH1)
        SF1 = (DF + (FREQ - F[K]) * Y) / ((FREQ - F[K])**2 + DF * DF)
        SF2 = (DF - (FREQ + F[K]) * Y) / ((FREQ + F[K])**2 + DF * DF)
        SUM = SUM + STR * (SF1 + SF2) * (FREQ / F[K])**2

    O2ABS = .5034E12 * SUM * PRESDA * TH**3 / 3.14159

    return O2ABS


@numba.jit(cache=True, nopython=True)
def ABSN2_IM10(T, P, F):
    '''
    ****ABSN2 = ABSORPTION COEFFICIENT DUE TO NITROGEN IN AIR (NEPER/KM)
    T = TEMPERATURE (K)
    P = PRESSURE (MB)
    F = FREQUENCY (GHZ)
    '''
    TH = 300. / T
    ALPHA = 6.4E-14 * P * P * F * F * TH**3.55

    return ALPHA


@numba.jit(cache=True, nopython=True)
# https://github.com/numba/numba/issues/2518
def TB_CALC_PL_IM10(T, tau, mu_s, freq):
    '''
    calculate brightness temperatures without scattering
    according to Simmer (94) pp. 87 - 91 (alpha = 1, no scattering)
    Planck/thermodynamic conform (28.05.03) # UL
    '''
    h = 6.6262e-34  # Planck constant
    kB = 1.3806e-23  # Boltzmann constant
    c_li = 2.997925*1e8  # Lichtgeschw.

    kmax = len(T)
    n_f = len(freq)

    # tau = np.float64(tau)

    # T = np.float64(T)
    mu = np.zeros(n_f) + mu_s
    freq_si = freq*1e9
    lamda_si = c_li/freq_si

    IN = np.zeros(n_f, dtype=np.float64) + 2.73
    IN = (2.*h*freq_si/(lamda_si**2.))*1./(np.exp(h*freq_si/(kB*IN))-1.)

    tau_top = np.zeros(n_f, dtype=np.float64)
    tau_bot = tau[kmax-2]
    for i in range(kmax-1):

        valid = 1
        if i > 0:
            tau_top = tau[kmax-2-i+1]
            tau_bot = tau[kmax-2-i]

        for ii in range(n_f):
            if tau_bot[ii] == tau_top[ii]:
                valid = 0
            if tau_bot[ii] < tau_top[ii]:
                valid = -1
        if valid == 0:
            print('warning, zero absorption coefficient')
        if valid == -1:
            print('warning, negative absorption coefficient')

        if valid == 1:
            delta_tau = tau_bot-tau_top
            A = np.ones(n_f, dtype=np.float64) - np.exp(-1*delta_tau/mu)
            B = delta_tau - mu + mu*np.exp(-1*delta_tau/mu)

            T_pl2 = (2.*h*freq_si/(lamda_si**2.))*1. / \
                (np.exp(h*freq_si/(kB*T[kmax-2-i]))-1)
            T_pl1 = (2.*h*freq_si/(lamda_si**2.))*1. / \
                (np.exp(h*freq_si/(kB*T[kmax-1-i]))-1)
            diff = (T_pl2 - T_pl1)/delta_tau
            IN = IN*np.exp(-1*delta_tau/mu) + T_pl1*A + diff*B

    TB = (h*freq_si/kB)*1./np.log((2*h*freq_si/(IN*lamda_si**2.))+1.)

    return TB


def doTests():
    '''
    results of these tests are compared with Uli's IDL routines.
    '''

    assert np.isclose(ABWVR98_IM10(
        10, 250, 700, np.array([35])), np.array([0.0252045]))
    assert np.isclose(ABSN2_IM10(250, 700, 53), 0.00016827562671213744)
    assert np.isclose(ABO2R98_IM10(250, 700, 10, 53), 0.159599435186749)

    tau, tau_wv, tau_o2 = TAU_CALC_IM10(
        np.array([10., 1000., 10000.]),        # height [m]
        np.array([300., 270., 250.]),        # Temp. [K]
        np.array([100000., 75000., 20000.]),        # press. [Pa]
        np.array([10e-3, 5e-3, 1e-3]),     # abs. hum. [kg m^-3]
        np.array([35.]),        # freq. [GHz]

    )
    assert np.all(np.isclose(tau, np.array([[0.06479182],
                                            [0.04500077]])))
    assert np.all(np.isclose(tau_wv, np.array([[0.04458111],
                                               [0.03022091]])))
    assert np.all(np.isclose(tau_o2, np.array([[0.01993734],
                                               [0.01457681]])))

    tau, tau_wv, tau_o2 = TAU_CALC_IM10(
        np.array([10., 1000., 10000.]),        # height [m]
        np.array([300., 270., 250.]),        # Temp. [K]
        np.array([100000., 75000., 20000.]),        # press. [Pa]
        np.array([10e-3, 5e-3, 1e-3]),     # abs. hum. [kg m^-3]
        np.array([53.]),        # freq. [GHz]
    )
    assert np.all(np.isclose(tau, np.array([[0.84145784],
                                            [0.62035005]])))
    assert np.all(np.isclose(tau_wv, np.array([[0.08091379],
                                               [0.05529443]])))
    assert np.all(np.isclose(tau_o2, np.array([[0.75991718],
                                               [0.56459001]])))

    TB, tau, tau_wv, tau_o2 = STP_IM10(
        # [m] states grid of T_final [K], p_final [Pa], q_final [kgm^-3]
        np.array([10., 1000., 10000.]),
        np.array([300., 270., 250.]),
        np.array([100000., 75000., 20000.]),
        np.array([10e-3, 5e-3, 1e-3]),
        10.,  # zenith angle of observation in deg.
        np.array([35., 53.]),  # frequency vector in GHz
    )
    assert np.all(np.isclose(TB, np.array([19.68133037, 156.13616292])))
    assert np.all(np.isclose(tau, np.array([[0.06479182, 0.84145784],
                                            [0.04500077, 0.62035005]])))
    assert np.all(np.isclose(tau_wv, np.array([[0.04458111, 0.08091379],
                                               [0.03022091, 0.05529443]])))
    assert np.all(np.isclose(tau_o2, np.array([[0.01993734, 0.75991718],
                                               [0.01457681, 0.56459001]])))

    heights = [
        0.0000000e+00, 4.2440624e+01, 8.5076721e+01, 1.2790133e+02, 
        1.7091106e+02, 2.1410199e+02, 2.5747302e+02, 3.0102405e+02, 
        3.4475641e+02, 3.8867044e+02, 4.3276773e+02, 4.7704846e+02, 
        5.2151447e+02, 5.6616852e+02, 6.1101276e+02, 6.5604932e+02, 
        7.0128076e+02, 7.4670978e+02, 7.9233929e+02, 8.3817181e+02, 
        8.8420935e+02, 1.0704741e+03, 1.2602533e+03, 1.4537362e+03, 
        1.6511360e+03, 1.8526676e+03, 2.0585374e+03, 2.2689570e+03, 
        2.4841409e+03, 2.7043225e+03, 2.9297371e+03, 3.1606392e+03, 
        3.3973149e+03, 3.6400652e+03, 3.8892119e+03, 4.1450962e+03, 
        4.8168247e+03, 5.5401792e+03, 6.3240845e+03, 7.1800732e+03, 
        8.1243667e+03, 9.1821113e+03, 1.0395719e+04, 1.1844845e+04, 
        1.2703082e+04, 1.3693662e+04, 1.4865961e+04, 1.6300934e+04, 
        1.8150615e+04, 2.0760561e+04, 2.4076291e+04, 2.5271123e+04, 
        2.6741920e+04, 2.8651969e+04, 3.1373289e+04, 3.3807105e+04, 
        3.6156855e+04, 3.7751129e+04, 3.9850242e+04, 4.2892234e+04
        ]
    temperature = [
        293.06253, 295.9907, 297.61807, 298.29483, 298.47858, 298.34824,  
        298.0353, 297.71616, 297.38647, 297.0542, 296.6928, 296.3217,  
        295.95053, 295.5693, 295.18384, 294.79764, 294.40634, 294.00082,  
        293.5864, 293.172, 292.75635, 291.04688, 289.47742, 288.29865,  
        287.69485, 286.70486, 285.5756, 284.28503, 282.8634, 281.38602,
        279.77945, 278.14996, 276.42813, 274.7154, 273.0224, 
        271.39532,  267.7346, 263.31265, 258.08643, 251.79128, 
        244.4248, 235.92027,  226.98349, 224.13881, 224.78397, 
        225.24701, 225.24701, 225.24701,  225.3689, 226.56195, 229.273, 
        230.22314, 231.79558, 233.58482,  236.5767, 242.67717, 248.45955, 
        253.52713, 258.5947, 266.24683
        ]
    pressure = [
        100000.,  99500.,  99000.,  98500.,  98000.,  97500.,  97000.,   
        96500.,  96000.,  95500.,  95000.,  94500.,  94000.,  93500.,   
        93000.,  92500.,  92000.,  91500.,  91000.,  90500.,  90000.,   
        88000.,  86000.,  84000.,  82000.,  80000.,  78000.,  76000.,   
        74000.,  72000.,  70000.,  68000.,  66000.,  64000.,  62000.,   
        60000.,  55000.,  50000.,  45000.,  40000.,  35000.,  30000.,   
        25000.,  20000.,  17500.,  15000.,  12500.,  10000.,   7500.,    
        5000.,   3000.,   2500.,   2000.,   1500.,   1000.,    700.,     
        500.,    400.,    300.,    200.
        ]
    humidity = [
        1.18251285e-02, 1.14939269e-02, 1.11975744e-02, 1.09878751e-02, 
        1.08402278e-02, 1.07536148e-02, 1.07092401e-02, 1.06639238e-02, 
        1.06170261e-02, 1.05705503e-02, 1.05287395e-02, 1.04884841e-02, 
        1.04482342e-02, 1.04073379e-02, 1.03661669e-02, 1.03249513e-02, 
        1.02846604e-02, 1.02469595e-02, 1.02108689e-02, 1.01747783e-02, 
        1.01378299e-02, 9.95743275e-03, 9.41002835e-03, 8.42119381e-03, 
        6.94509782e-03, 5.98434173e-03, 5.21718571e-03, 4.67937114e-03, 
        4.31687059e-03, 4.02891682e-03, 3.81084811e-03, 3.57917766e-03, 
        3.26925074e-03, 2.88613979e-03, 2.34445557e-03, 1.81542535e-03, 
        7.76122033e-04, 6.57636556e-04, 2.95107893e-04, 1.55331218e-04, 
        9.08064612e-05, 6.24438544e-05, 4.23682905e-05, 2.08727743e-05, 
        1.37270645e-05, 9.99999975e-06, 9.99999975e-06, 9.99999975e-06, 
        9.99999975e-06, 9.99999975e-06, 9.99999975e-06, 9.99999975e-06, 
        9.99999975e-06, 9.99999975e-06, 9.99999975e-06, 9.99999975e-06, 
        9.99999975e-06, 9.99999975e-06, 9.99999975e-06, 9.99999975e-06
        ]
    zenithAngle = 0
    frequencies = [51.26, 52.28, 53.86, 54.94, 56.66, 57.3, 58.]
    TB, tau, tau_wv, tau_o2 = STP_IM10(
        heights, temperature, pressure, humidity, zenithAngle, frequencies)
    TBref = np.array([
        112.38722248, 155.71521178, 259.08914502, 289.34402775,
        295.16305358, 295.67990855, 295.98309774
                      ])
    assert np.all(np.isclose(TB, TBref))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import numpy as np 
import random
import scipy.interpolate as interp
import re

from typing import Sequence

# ---------------------------------------------------------------------
# old functions

standards = pd.read_hdf(r'C:/Users/juand/Research/h5_files/standards_230801.h5').reset_index(drop=True)
# standard_types = list(range(15,40))
# flux_standards = [standards.interpolated_flux[type-10] for type in standard_types]
# flux_standards = [standards.interpolated_flux[i] for i in range(len(standards))]
wavegrid = standards["WAVE"].iloc[0]
wavegrid_list = list(wavegrid)
STANDARDS = {
    "WAVE": wavegrid,
    "SPT": standards["SPT"],
    "FLUX": standards["FLUX"],
    "UNC": standards["UNCERTAINTY"],
}
# wavegrid = STANDARDS["WAVE"]

# ------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------
BINARIES={}
# ------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------

def interpolate_flux_wave(
    wave: Sequence, flux: Sequence, wgrid: Sequence, verbose: bool = True
):
    """
    filterMag function requires interpolation to different wavelengths.
    Function to interpolate the flux from the stars to the wavegrid we are working on.
    Parameters
    ----------
    wave : Sequence
        An array specifying wavelength in units of microns of the given star.
    flux : Sequence
        An array specifying flux density in f_lambda units of the given star.
    wgrid : Sequence
        An array specifying wavelength in units of microns on which the star
        will be interpolated.
    Returns
    -------
    interpolated_flux : Sequence
        An array with the interpolated flux.
    """
    f = interp.interp1d(wave, flux, assume_sorted=False, fill_value=0.0)
    return f(wgrid)

# ------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------


def initialize_binaries():
    bina_df = pd.read_hdf(r'C:/Users/juand/Research/h5_files/spectral_templates_aug3_normalized.h5', key='binaries')
    b_wavegrid = np.array(pd.read_hdf(r'C:/Users/juand/Research/h5_files/spectral_templates_aug3_normalized.h5', key='wavegrid'))
    interpol_flux=[]
    for j in range(len(bina_df)):
        a=[]
        for i in range(409):
            a.append(bina_df["flux_" + str(i)][j])
        interpol_flux.append(a)
    bina_df["interpol_flux"]=interpol_flux
    bina_df = bina_df.loc[bina_df['primary_type']<=bina_df['secondary_type']]
    bina_df = bina_df.reset_index(drop=True)
    new_wave=wavegrid
    new_wave[-1]=b_wavegrid[-1]
    fluxlist=[]
    for i in range(len(bina_df)):
        fluxi=bina_df['interpol_flux'][i]
        nfluxi = interpolate_flux_wave(b_wavegrid, fluxi, wgrid=new_wave)
        fluxlist.append(nfluxi)
    bina_df['FLUX']=fluxlist    
    BINARIES = {
        "WAVE": wavegrid,
        "PRIM": bina_df["primary_type"],
        "SECO": bina_df["secondary_type"],
        "FLUX": bina_df["FLUX"],
    }
    return


# ------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------

def measureSN(wave, flux, unc, rng=[1.2, 1.35], verbose=True):
    """
    Measures the signal-to-noise of a spectrum over a specified wavelength range
    Parameters
    ----------
    wave : list or numpy array of floats
                    An array specifying wavelength in units of microns
    flux : list or numpy array of floats
                    An array specifying flux density in f_lambda units
    unc : list or numpy array of floats
                    An array specifying uncertainty in the same units as flux
    rng : 2-element list of floats, default = [1.2,1.35]
                    Specifies the range over which S/N is measured
    Returns
    -------
    float
        Median signal-to-noise value in the specified range
    Examples
    --------
    >>> wave = np.linspace(1,3,100)
    >>> flux = np.random.normal(5,0.2,100)
    >>> unc = flux*0.01
    >>> measureSN(wave,flux,unc)
    0.01
    """

    idx = np.where((wave <= rng[1]) & (wave >= rng[0]))
    return np.nanmedian(flux[idx] / unc[idx])

def typeToNum(inp):
    """
    Converts between string and numeric spectral types, with the option of specifying the class prefix/suffix and uncertainty tags
    Parameters
    ----------
        Spectral type to convert. Can convert a number or a string from 0.0 (K0) and 49.0 (Y9).
    Returns
    -------
        The number or string of a spectral type
    Example
    -------
        >>> print splat.typeToNum(30)
            T0.0
        >>> print splat.typeToNum('T0.0')
            30.0
        >>> print splat.typeToNum(50)
            Spectral type number must be between 0 (K0) and 49.0 (Y9)
            nan
    """

    spletter = "KMLTY"

    if isinstance(inp, list):
        raise ValueError(
            "\nInput to typeToNum() must be a single element (string or number)"
        )

    elif isinstance(inp, str):
        inp = inp.split("+/-")[0]
        inp = inp.replace("...", "").replace(" ", "")
        sptype = re.findall("[{}]".format(spletter), inp.upper())
        outval = 0.0
        outval = spletter.find(sptype[0]) * 10.0
        spind = inp.find(sptype[0]) + 1
        if inp.find(".") < 0:
            outval = outval + float(inp[spind])
        else:
            try:
                outval = outval + float(inp[spind : spind + 3])
                spind = spind + 3
            except:
                print(
                    "\nProblem converting input type {} to a numeric type".format(
                        inp
                    )
                )
                outval = np.nan
        return outval
    
    elif type(inp) == int or float:
        spind = int(abs(inp / 10.0))
        if spind < 0 or spind >= len(spletter):
            print(
                "Spectral type number must be between 0 ({}0) and {} ({}9)".format(
                    spletter[0], len(spletter) * 10.0 - 1.0, spletter[-1]
                )
            )
            print("N/A")
        spdec = np.around(inp, 1) - spind * 10.0
        return "{}{:3.1f}".format(spletter[spind], spdec)

    else:
        print(
            "\nWarning: could not recognize format of spectral type {}\n".format(
                inp
            )
        )
        return inp
    
def addNoise(flux, unc, scale=1.0):
    """
    Resamples data to add noise, scaled according to input scale factor (scale > 1 => increased noise)
    Parameters
    ----------
    flux : list or numpy array of floats
                    An array specifying flux density in f_lambda units
    unc : list or numpy array of floats
                    An array specifying uncertainty in the same units as flux
    scale : float, default = 1.
                    Scale factor to scale uncertainty array; scale > 1 reduces signal-to-noise
    Returns
    -------
    list or numpy array of floats
            flux with noise added
    list or numpy array of floats
            scaled uncertainty
    Examples
    --------
    >>> wave = np.linspace(1, 3, 100)
    >>> flux = np.random.normal(5, 0.2, 100)
    >>> unc = flux * 0.01
    >>> nflux, nunc = addNoise(flux, unc, scale=5.)
    """
    sunc = unc * scale
    if scale > 1.0:
        nunc = sunc
    else:
        nunc = unc
    nflux = np.random.normal(flux, sunc)  # random number
    return nflux, nunc

def fast_classify(
    wave,
    flux,
    unc,
    fit_range=[0.9, 2.4],
    standards=STANDARDS,
    telluric=False,
    method="full",
):
    """
    This function was aded by Juan Diego to replace the previousfast classify
    The function uses the mathematical methd used by Bardalez 2014 to classify the stars comparing them to standards
    Parameters
    ----------
    wave : list or numpy array of floats
                An array specifying wavelength in microns
    flux : list or numpy array of floats
                An array specifying flux density in f_lambda units
    uncertainties : list or numpy array of floats
                An array specifying uncertainty in the same units as flux
    standards : dict
                Dictionary containind 1D array 'WAVE', 1D array 'SPT', and Nx1D array 'FLUX'
                it is assumed 'WAVE' in this array is same as input spectrum
    fit_range : list or numpy array of 2 floats
                Default = [0.9, 2.4]
                An array specifying the wavelength values between which the function will be classified
    telluric : bool
                Default = True
                A value that defines whether the function will mask out regions where there is telluric/atmospheric absorbtion
    method : string
                Default: 'full'
                When set to method='kirkpatrick', the fit range is adjusted to [0.9,1.4]
    Returns
    -------
    float
            Numerical spectral type of classification, with 15 = M5, 25 = L5, 35 = T5, etc
    Example
    -------
    >>> flux_21 = standards.interpolated_flux[21-10]
    >>> noise_21 = standards.interpolated_noise[21-10]
    >>> fast_classify(wavegrid, flux_21, noise_21)
    21
    """
    if method == "kirkpatrick":
        fit_range = [0.9, 1.4]
    elif method == "full":
        fit_range = [0.9, 2.4]
    else:
        pass

    w = np.where(np.logical_and(wave >= fit_range[0], wave <= fit_range[1]))[0]

    scales, chi = [], []

    # weights = np.array([wavegrid[1]-wavegrid[0]] + [(wavegrid[i]-wavegrid[i-1])/2 + (wavegrid[i+1]-wavegrid[i])/2 for i in w[1:-1]] + [wavegrid[-1]-wavegrid[-2]])
    # weights = np.array([wavegrid[1]-wavegrid[0]] + [(wavegrid[i+1]-wavegrid[i-1])/2 for i in w[1:-1]] + [wavegrid[-1]-wavegrid[-2]])
    # weights = np.array([wave[1]-wave[0]] + list((wave[2:]-wave[:-2])/2) + [wave[-1]-wave[-2]])
    weights = np.ones(len(wave))

    if telluric == True:
        msk = np.ones(len(weights))
        msk[
            np.where(
                np.logical_or(
                    np.logical_and(wavegrid > 1.35, wavegrid < 1.42),
                    np.logical_and(wavegrid > 1.8, wavegrid < 1.95),
                )
            )
        ] = 0
        weights = weights * msk

    # Loop through standards
    for std in standards["FLUX"]:
        scale = np.nansum(weights * (flux * std) / (unc**2)) / np.nansum(
            (weights * std**2) / (unc**2)
        )
        scales.append(scale)
        chisquared = np.nansum(
            weights * ((flux - (std * scales[-1])) ** 2) / (unc**2)
        )
        chi.append(chisquared)
        
    if standards==BINARIES:
        return standards["PRIM"][np.argmin(chi)], standards["SECO"][np.argmin(chi)]
    else:
        return standards["SPT"][np.argmin(chi)]
    
def normalize(wave, flux, unc, rng=[1.2, 1.35], method="median"):
    """
    Normalizes the spectrum of an object over a given wavelength range
    Parameters
    ----------
    wave : list or numpy array of floats
                    An array specifying wavelength in units of microns
    flux : list or numpy array of floats
                    An array specifying flux density in f_lambda units
    unc : list or numpy array of floats
                    An array specifying uncertainty in the same units as flux
    rng : 2-element list of floats, default = [1.2,1.35]
                    Specifies the range over which normalization is computed
    method : str, default = 'median'
                    Specifies the method by which normalization is determined; options are:
                    * 'median': compute the median value within rng
                    * 'max': compute the maximum value within rng
    Returns
    -------
    list or numpy array of floats
            normalized flux
    list or numpy array of floats
            normalized uncertainty
    Examples
    --------
    >>> wave = np.linspace(1,3,100)
    >>> flux = np.random.normal(5,0.2,100)
    >>> unc = flux*0.01
    >>> nflux, nunc = normalize(wave,flux,unc)
    """
    idx = np.where((wave <= rng[1]) & (wave >= rng[0]))
    n_flux = flux / np.nanmax(flux[idx])
    n_unc = unc / np.nanmax(flux[idx])
    return n_flux, n_unc

# ------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------


def combine_two_spex_spectra(flux1, unc1, flux2, unc2, name1="", name2=""):
    """
    A function that combines the spectra of the two given stars
    Parameters
    ----------
        flux1 : list or array of floats
                    An array specifying flux density of the first star in f_lambda units
        unc1 : list or array of floats
                    An array specifying noise of the first star in f_lambda units
        flux2 : list or array of floats
                    An array specifying flux density of the second star in f_lambda units
        unc2 : list or array of floats
                    An array specifying noise of the second star in f_lambda units
    Returns
    -------
        dictionary :
            "primary_type": spectral type of the first star,
            "secondary_type": spectral type of the second star,
            "system_type": spectral type of the combined spectrum,
            "system_interpolated_flux": array specifying flux density of the combined spectrum
            "system_interpolated_noise": array specifying noise of the combined spectrum
            "difference_spectrum": aray specifying the difference with the standard of the same type,
    Examples
    --------
    """

    # Classify the given spectra
    spt1 = fast_classify(wavegrid, flux1, unc1)
    spt2 = fast_classify(wavegrid, flux2, unc2)

    # # get magnitudes of types
    # absj1 = typeToMag(spt1)[0]
    # absj2 = typeToMag(spt2)[0]

    # # Calibrate flux
    # flux1, unc1 = fluxCalibrate(flux1, unc1, "2MASS_J", absj1)
    # flux2, unc2 = fluxCalibrate(flux2, unc2, "2MASS_J", absj2)

    # Create combined spectrum
    flux3 = flux1 + flux2
    unc3 = unc1 + unc2

    # Classify Result
    spt3 = fast_classify(wavegrid, flux3, unc3)

    # get standard
    spt3_num = typeToNum(spt3)
    spt3_num = round(spt3_num)
    type_position = []
    for position in range(len(STANDARDS['SPT'])):
        if int(typeToNum(STANDARDS['SPT'][position]))==spt3_num:
            type_position.append(position)
    type_position = type_position[0]
    flux_standard = STANDARDS["FLUX"][type_position]
    unc_standard = STANDARDS["UNC"][type_position]

    # normalize
    flux1, unc1 = normalize(wavegrid, flux1, unc1)
    flux2, unc2 = normalize(wavegrid, flux2, unc2)
    flux3, unc3 = normalize(wavegrid, flux3, unc3)

    # diff
    diff = flux_standard - flux3

    if isinstance(name1, str) & isinstance(name2, str):
        name = name1 + "+" + name2

    return {
        "primary_type": spt1,
        "secondary_type": spt2,
        "system_type": spt3,
        "system_interpolated_flux": interpolate_flux_wave(
            wavegrid, flux3, wavegrid
        ).flatten(),
        "system_interpolated_noise": interpolate_flux_wave(
            wavegrid, unc3, wavegrid
        ).flatten(),
        "difference_spectrum": interpolate_flux_wave(
            wavegrid, np.abs(diff), wavegrid
        ).flatten(),
        "name": name,
    }





def _addstars(df, target, mintype='', maxtype=38, undersample=False, undersample_drop='random', flux_col_name='FLUX'):
    """
    Creates new stars by adding noise to the spectrum and distributes them equally
    Parameters
    ----------
    df : pandas dataframe
                    pandas table containing the following columns: ['FLUX', 'UNCERTAINTY', 'J_SNR', 'SPT', 'WAVE']
    target : float
                    desired number of stars per type
    
    mintype : float, default=''
                    the default makes it be the minimum type in the dataframe
                    desired smaller type number
    
    maxtype : float, default=''
                    the default makes it be the maximum type in the dataframe
                    desired larger type number
    undersample : bool, default = False
                    cannot be used yet
                    limit your number of stars of each type to a certain number by undersampling
    
    undersample : string, default = 'random'
                    other options are 'lowest', 'highest'
                    cannot be used yet
                    specify which stars to drop when undersampling: random, lowest (lowest snr), highest (highest snr)
    Returns
    -------
    pandas dataframe
    """
    # there could be an undersampling option if you want to limit your number of stars of each type to a certain number
    # could also specify which ones to drop: random, lowest (lowest snr), highest (highest snr)
    # to be implemented

    df = df.reset_index(drop=True)

    if mintype=='':
        mintype=int(min(df.SPT_NUM))
    if maxtype=='':
        maxtype=int(max(df.SPT_NUM))
    typesrange = range(mintype,maxtype+1)

    new_df = df.loc[df['SPT_NUM']<(maxtype+1)*np.ones(len(df))]
    drop_unc = []
    for position, uncertainty_i in enumerate(np.array(new_df['UNCERTAINTY'])):
        if np.any(uncertainty_i<0)|np.any(np.isnan(uncertainty_i))|np.any(np.isinf(uncertainty_i)):
            drop_unc.append(position)
    new_df = new_df.drop(drop_unc).reset_index(drop=True)

    for spt in list(typesrange):
        singles_type = new_df.loc[new_df['SPT_NUM']==spt*np.ones(len(new_df))]
        singles_type = singles_type.reset_index(drop=True)

        # high snr
        singles_snr = singles_type.loc[singles_type['J_SNR']>=100*np.ones(len(singles_type))]
        singles_snr = singles_snr.reset_index(drop=True)
        higsnrstars = singles_snr    
        have = len(higsnrstars)
        if have>0:
            need = target-have
            # if have>0:
            while need>0:
                star = random.randint(0, have-1)
                flux = singles_snr[flux_col_name][star]
                unc = singles_snr['UNCERTAINTY'][star]
                mult = singles_snr['J_SNR'][star]/100
                mult = mult -1
                noisescale = random.random()*(mult) + 1                 
                nflux, nunc = addNoise(flux, unc, scale=noisescale)
                nan_and_zeros = (len(nunc)-np.count_nonzero(nunc)) + (len(nunc)-np.count_nonzero(~np.isnan(nunc)))
                nunc = np.abs(nunc)
                snr = measureSN(singles_snr.WAVE[star], nflux, nunc)
                while (nan_and_zeros>0)&(snr<100):
                    noisescale = random.random()*(mult) + 1
                    nflux, nunc = addNoise(flux, unc, scale=noisescale)
                    nan_and_zeros = (len(nunc)-np.count_nonzero(nunc)) + (len(nunc)-np.count_nonzero(~np.isnan(nunc)))
                    nunc = np.abs(nunc)
                    snr = measureSN(singles_snr.WAVE[star], nflux, nunc)
                new_df.loc[len(new_df.index)] = [nflux, nunc, snr, 'hig', singles_snr.WAVE[star], singles_snr.SPT[star], singles_snr.SPT_NUM[star]] 
                need -= 1
        
        # mid snr
        singles_snr = singles_type.loc[singles_type['J_SNR']>=50*np.ones(len(singles_type))]
        singles_snr = singles_snr.reset_index(drop=True)
        midsnrstars = singles_snr.loc[singles_snr['J_SNR']<100*np.ones(len(singles_snr))]  
        have = len(midsnrstars)
        if have>0:
            need = target-have
            # if have>0:
            while need>0:
                star = random.randint(0, have-1)
                flux = singles_snr[flux_col_name][star]
                unc = singles_snr['UNCERTAINTY'][star]
                start = singles_snr['J_SNR'][star]/100
                if start<1:
                    start=1
                finish = singles_snr['J_SNR'][star]/50
                noisescale = random.random()*(finish - start) + start
                nflux, nunc = addNoise(flux, unc, scale=noisescale)
                nan_and_zeros = (len(nunc)-np.count_nonzero(nunc)) + (len(nunc)-np.count_nonzero(~np.isnan(nunc)))
                nunc = np.abs(nunc)
                snr = measureSN(singles_snr.WAVE[star], nflux, nunc)
                while (nan_and_zeros>0)&(snr<50)&(snr>100):
                    noisescale = random.random()*(finish - start) + start
                    nflux, nunc = addNoise(flux, unc, scale=noisescale)
                    nan_and_zeros = (len(nunc)-np.count_nonzero(nunc)) + (len(nunc)-np.count_nonzero(~np.isnan(nunc)))
                    nunc = np.abs(nunc)
                    snr = measureSN(singles_snr.WAVE[star], nflux, nunc)
                new_df.loc[len(new_df.index)] = [nflux, nunc, snr, 'mid', singles_snr.WAVE[star], singles_snr.SPT[star], singles_snr.SPT_NUM[star]] 
                need -= 1
        
        # low snr
        singles_snr = singles_type
        singles_snr = singles_snr.reset_index(drop=True)    
        lowsnrstars = singles_snr.loc[singles_snr['J_SNR']<50*np.ones(len(singles_snr))]  
        have = len(lowsnrstars)
        if have>0:
            need = target-have
            # if have>0:
            while need>0:
                star = random.randint(0, have-1)
                flux = singles_snr[flux_col_name][star]
                unc = singles_snr['UNCERTAINTY'][star]
                start = singles_snr['J_SNR'][star]/50
                if start<1:
                    start=1
                minimum = random.random()*(20)
                finish = singles_snr['J_SNR'][star]/minimum
                noisescale = random.random()*(finish - start) + start
                if finish<1:
                    noisescale=1
                nflux, nunc = addNoise(flux, unc, scale=noisescale)
                nan_and_zeros = (len(nunc)-np.count_nonzero(nunc)) + (len(nunc)-np.count_nonzero(~np.isnan(nunc)))
                nunc = np.abs(nunc)
                snr = measureSN(singles_snr.WAVE[star], nflux, nunc)
                while (nan_and_zeros>0)&(snr<0)&(snr>50):
                    minimum = random.random()*(25)
                    finish = singles_snr['J_SNR'][star]/minimum
                    noisescale = random.random()*(finish - start) + start
                    nflux, nunc = addNoise(flux, unc, scale=noisescale)
                    nan_and_zeros = (len(nunc)-np.count_nonzero(nunc)) + (len(nunc)-np.count_nonzero(~np.isnan(nunc)))
                    nunc = np.abs(nunc)
                    snr = measureSN(singles_snr.WAVE[star], nflux, nunc)
                new_df.loc[len(new_df)] = [nflux, nunc, snr, 'low', singles_snr.WAVE[star], singles_snr.SPT[star], singles_snr.SPT_NUM[star]] 
                need -= 1
    new_df = new_df.loc[new_df['SPT_NUM']>=mintype*np.ones(len(new_df))]
    new_df = new_df.loc[new_df['SPT_NUM']<=maxtype*np.ones(len(new_df))]
                
    return(new_df)

def addstars(df, target, mintype='', maxtype=38, flux_col_name='FLUX'):
    """
    Creates new stars by adding noise to the spectrum and distributes them equally and makes sure there are no nans
    Relies on addstars_1
    Parameters
    ----------
    df : pandas dataframe
                    pandas table containing the following columns: ['FLUX', 'UNCERTAINTY', 'J_SNR', 'SPT', 'WAVE']
    target : float
                    desired number of stars per type
    
    mintype : float, default=''
                    the default makes it be the minimum type in the dataframe
                    desired smaller type number
    
    maxtype : float, default=''
                    the default makes it be the maximum type in the dataframe
                    desired larger type number
    Returns
    -------
    pandas dataframe
    """
    df_new = _addstars(df, target=target, mintype=mintype,maxtype=maxtype)
    while (len(df_new)-len(df_new.dropna()))>0:
        df_new = df_new.dropna()
        df_new = df_new.reset_index(drop=True)
        df_new = _addstars(df_new, target=target, mintype=mintype, maxtype=maxtype)
    df_new = df_new.reset_index(drop=True)
    return df_new


def _binaryCreation_hig(singles_df, target, snr_range='hig', fluxSeparate=False, difference=False):

    """
    Creates binaries out of single stars by combining the spectra and adding noise and distributes them equally
    Parameters
    ----------
    singles_df : pandas dataframe of single stars
                    pandas table containing the following columns: ['FLUX', 'UNCERTAINTY', 'J_SNR', 'SPT', 'WAVE', 'SPT_NUM', 'SNR_CLASS']
    target : float
                    desired number of combinations per type
    
    snr_range : float, default='low'
                    desired snr for the binaries
                    low is from 0-50
                    mid is from 50-100
                    hig is larger than 100
    fluxSeparate : bool, default = False
                    separate the flux in each individual flux value per column
                    recommended: True
                    it allows to use the created dataframe to make a multioutput regressor
    Returns
    -------
    pandas dataframe
    """

    uppersnr = 100000
    lowersnr = 100
    dataframe = singles_df.loc[singles_df['J_SNR']>=lowersnr*np.ones(len(singles_df))]
    dataframe.reset_index(drop=True, inplace=True)
    

    fluxes=[]
    noises=[]
    primaries=[]
    secondaries=[]
    snr_list=[]
    differences=[]
    for j in range(16,40):
        print(j)
        if len(dataframe.loc[dataframe['SPT_NUM'] == j]) == 0:
            continue    # continue here

        for k in range(j,40):
            if len(dataframe.loc[dataframe['SPT_NUM'] == k]) == 0:
                    continue
            
            for i in range(0,target):
                nanvalues=1
                snr3=-1
                while (nanvalues!=0)|(lowersnr>snr3)|(uppersnr<snr3)|(snr3>uppersnr):
                    m1 = random.randint(0,len(dataframe.loc[dataframe['SPT_NUM'] == j])-1)
                    n1 = random.randint(0,len(dataframe.loc[dataframe['SPT_NUM'] == k])-1)
                    flux1 = np.array(dataframe.loc[dataframe['SPT_NUM'] == j].reset_index(drop=True)['FLUX'][m1])
                    unc1  = np.array(dataframe.loc[dataframe['SPT_NUM'] == j].reset_index(drop=True)['UNCERTAINTY'][m1])
                    flux2 = np.array(dataframe.loc[dataframe['SPT_NUM'] == k].reset_index(drop=True)['FLUX'][n1])
                    unc2  = np.array(dataframe.loc[dataframe['SPT_NUM'] == k].reset_index(drop=True)['UNCERTAINTY'][n1])

                    flux1, unc1 = addNoise(flux1, unc1, scale=1)
                    flux2, unc2 = addNoise(flux2, unc2, scale=1)

                    combstar_dic = combine_two_spex_spectra(flux1, unc1, flux2, unc2)
                    flux3 = np.array(combstar_dic["system_interpolated_flux"])
                    unc3  = np.array(combstar_dic["system_interpolated_noise"])
                    snr3 = measureSN(dataframe['WAVE'][0], flux3, unc3)
                    if uppersnr<snr3:
                        start = snr3/uppersnr
                        if start<1:
                            start=1
                        finish = random.random()*(snr3-uppersnr)+uppersnr*random.random()*random.random()*random.random()*random.random()*random.random()
                        if lowersnr!=0:
                            finish = snr3/lowersnr
                        noisescale = np.abs(random.random()*(finish-start)+start*random.random())
                        flux3, unc3 = addNoise(flux3,unc3, scale=noisescale)
                        snr3 = measureSN(dataframe['WAVE'][0], flux3, unc3)

                    # check nans
                    nanvalues=np.sum(np.isnan(flux3)) + np.sum(np.isnan(unc3)) + np.sum(np.isnan(snr3))
                fluxes.append(flux3)
                noises.append(unc3)
                primaries.append(j)
                secondaries.append(k)
                snr_list.append(snr3)
                if difference==True:
                    diff3 = np.array(combstar_dic["difference_spectrum"])
                    differences.append(diff3)
            
    d = {"system_interpolated_flux": fluxes,
        "system_interpolated_noise": noises,
        "primary_type": primaries,
        "secondary_type": secondaries,
        "J_SNR": snr_list,
        "SNR_CLASS": snr_range,
        "WAVE": [dataframe['WAVE'][0] for i in range(len(fluxes))]
        }
    if difference==True:
        d["difference_spectrum"]=differences
    BinDF = pd.DataFrame(d)

    if fluxSeparate==True:
        flux_cols_dic = {}
        for j in range(len(BinDF['system_interpolated_flux'][0])):
            fluxcol=[]
            for i in range(len(BinDF)):
                fluxcol.append(BinDF['system_interpolated_flux'][i][j])
            fluxname='flux_'+str(j)
            flux_cols_dic[fluxname] = fluxcol
        flux_cols_df = pd.DataFrame(flux_cols_dic)
        BinDF = pd.concat([BinDF, flux_cols_df], axis=1)

        if difference==True:
            diff_cols_dic = {}
            for j in range(len(BinDF['difference_spectrum'][0])):
                diffcol=[]
                for i in range(len(BinDF)):
                    diffcol.append(BinDF['difference_spectrum'][i][j])
                diffname='diff_'+str(j)
                diff_cols_dic[diffname] = diffcol
            diff_cols_df = pd.DataFrame(diff_cols_dic)
            BinDF = pd.concat([BinDF, diff_cols_df], axis=1)
    
    return BinDF


def binaryCreation(singles_df, target, snr_range='low', fluxSeparate=False, difference=False):

    """
    Creates binaries out of single stars by combining the spectra and adding noise and distributes them equally
    Parameters
    ----------
    singles_df : pandas dataframe of single stars
                    pandas table containing the following columns: ['FLUX', 'UNCERTAINTY', 'J_SNR', 'SPT', 'WAVEGRID', 'SPT_NUM', 'SNR_CLASS']
    target : float
                    desired number of combinations per type
    
    snr_range : float, default='low'
                    desired snr for the binaries
                    low is from 0-50
                    mid is from 50-100
                    hig is larger than 100
    fluxSeparate : bool, default = False
                    separate the flux in each individual flux value per column
                    recommended: True
                    it allows to use the created dataframe to make a multioutput regressor
    Returns
    -------
    pandas dataframe
    """

    buildingblocks = 50
    if snr_range=='low':
        uppersnr = 50
        lowersnr = 0
        dataframe = singles_df.loc[singles_df['J_SNR']>=lowersnr*np.ones(len(singles_df))]
        dataframe.reset_index(drop=True, inplace=True)
    elif snr_range=='mid':
        uppersnr = 100
        lowersnr = 50
        dataframe = singles_df.loc[singles_df['J_SNR']>=lowersnr*np.ones(len(singles_df))]
        dataframe.reset_index(drop=True, inplace=True)
    elif snr_range=='hig':
        return _binaryCreation_hig(singles_df,target)
    else:
        return print('Not a valid entry for the snr_range. Chose between "low", "mid", "hig".')    
    
    smalltarget = int(target/buildingblocks)
    step = (uppersnr-lowersnr)/buildingblocks
    up = step + lowersnr
    lo = lowersnr

    fluxes=[]
    noises=[]
    primaries=[]
    secondaries=[]
    snr_list=[]
    differences=[]
    for stp in range(buildingblocks):
        upper = up + step*stp
        lower = lo + step*stp
        print(upper)
        for j in range(16,40):
            if upper>99:
                print(j)
            
            if len(dataframe.loc[dataframe['SPT_NUM'] == j]) == 0:
                continue    # continue here

            for k in range(j,40):
                if len(dataframe.loc[dataframe['SPT_NUM'] == k]) == 0:
                        continue
            
                for i in range(0,smalltarget):
                    nanvalues=1
                    snr3=-1

                    while (nanvalues!=0)|(lower>snr3)|(upper<snr3):
                        # get a random star of each type we are looking for
                        m1 = random.randint(0,len(dataframe.loc[dataframe['SPT_NUM'] == j])-1)
                        n1 = random.randint(0,len(dataframe.loc[dataframe['SPT_NUM'] == k])-1)
                        flux1 = np.array(dataframe.loc[dataframe['SPT_NUM'] == j].reset_index(drop=True)['FLUX'][m1])
                        unc1  = np.array(dataframe.loc[dataframe['SPT_NUM'] == j].reset_index(drop=True)['UNCERTAINTY'][m1])
                        flux2 = np.array(dataframe.loc[dataframe['SPT_NUM'] == k].reset_index(drop=True)['FLUX'][n1])
                        unc2  = np.array(dataframe.loc[dataframe['SPT_NUM'] == k].reset_index(drop=True)['UNCERTAINTY'][n1])

                        flux1, unc1 = addNoise(flux1, unc1, scale=1)
                        flux2, unc2 = addNoise(flux2, unc2, scale=1)

                        combstar_dic = combine_two_spex_spectra(flux1, unc1, flux2, unc2)
                        flux3 = np.array(combstar_dic["system_interpolated_flux"])
                        unc3  = np.array(combstar_dic["system_interpolated_noise"])
                        snr3 = measureSN(dataframe['WAVE'][0], flux3, unc3)
                        if upper<snr3:
                            start = snr3/upper
                            if start<1:
                                start=1
                            finish = random.random()*(snr3-upper)+upper*random.random()*random.random()*random.random()
                            if lower!=0:
                                finish = snr3/lower
                            noisescale = np.abs(random.random()*(finish-start)+start*random.random())
                            flux3, unc3 = addNoise(flux3,unc3, scale=noisescale)
                            snr3 = measureSN(dataframe['WAVE'][0], flux3, unc3)

                        # check nans
                        nanvalues=np.sum(np.isnan(flux3)) + np.sum(np.isnan(unc3)) + np.sum(np.isnan(snr3))
                    fluxes.append(flux3)
                    noises.append(unc3)
                    primaries.append(j)
                    secondaries.append(k)
                    snr_list.append(snr3)
                    if difference==True:
                        diff3 = np.array(combstar_dic["difference_spectrum"])
                        differences.append(diff3)
            
    d = {"system_interpolated_flux": fluxes,
        "system_interpolated_noise": noises,
        "primary_type": primaries,
        "secondary_type": secondaries,
        "J_SNR": snr_list,
        "SNR_CLASS": snr_range,
        "WAVE": [dataframe['WAVE'][0] for i in range(len(fluxes))]
        }
    if difference==True:
        d["difference_spectrum"]=differences
    BinDF = pd.DataFrame(d)

    if fluxSeparate==True:

        print('Flux Separation:')    

        flux_cols_dic = {}
        for j in range(len(BinDF['system_interpolated_flux'][0])):
            fluxcol=[]
            for i in range(len(BinDF)):
                fluxcol.append(BinDF['system_interpolated_flux'][i][j])
            fluxname='flux_'+str(j)
            flux_cols_dic[fluxname] = fluxcol
        flux_cols_df = pd.DataFrame(flux_cols_dic)
        BinDF = pd.concat([BinDF, flux_cols_df], axis=1)

        if difference==True:

            print('Flux Difference:')

            diff_cols_dic = {}
            for j in range(len(BinDF['difference_spectrum'][0])):
                diffcol=[]
                for i in range(len(BinDF)):
                    diffcol.append(BinDF['difference_spectrum'][i][j])
                diffname='diff_'+str(j)
                diff_cols_dic[diffname] = diffcol
            diff_cols_df = pd.DataFrame(diff_cols_dic)
            BinDF = pd.concat([BinDF, diff_cols_df], axis=1)
    
    return BinDF


def binary_multiOutput_classifier(bin_df, telluric=False, difference=False, n_estimators=100, test_size=0.30, max_depth=15, min_samples_leaf=2, min_samples_split=5,n_jobs=1, traindata=False, testdata=False, shape=False, shuffle=True, random_state=42):
    """
    Creates a multi output random forest model
    Parameters
    ----------
    bin_df : pandas dataframe
                    pandas table containing the following columns: ['primary_type','secondary_type','system_interpolated_flux','system_interpolated_noise','snr','SNR_CLASS','flux_0','flux_1', ... , 'flux_408']
    difference : bool, default = False
                    create the random forest model with the difference flux (binary_flux - standard_flux)
                    Warning! This input doubles the parameters in the random forest. 
    traindata : bool, default = False
                    optional output of a dictionary with the train data
    testdata : bool, default = False
                    optional output of a dictionary with the test data
    shape : bool, default = False
                    option to print the shape of the input data to the RF model
    Returns
    -------
    sklearn.multioutput.MultiOutputRegressor
    optional: dictionary
    Examples
    --------
    >>> MultiOutputRegressor_Create(bin_df)
    >>> RFmodel, traintest_data = MultiOutputRegressor_Create(bin_df, traindata=True, testdata=True)
    """
    feats = list(bin_df.columns)
    feats.remove('primary_type')
    feats.remove('secondary_type')
    feats.remove('system_interpolated_flux')
    feats.remove('system_interpolated_noise')
    feats.remove('J_SNR')
    feats.remove('SNR_CLASS')
    if 'WAVE' in feats:
        feats.remove('WAVE')
    if difference==True:
        feats.remove('difference_spectrum')
    
    wvgrd = bin_df['WAVE'][0]
    if telluric==True:
        telluric_mask = list(np.where(np.logical_or(np.logical_and(wvgrd > 1.35,wvgrd < 1.42), np.logical_and(wvgrd > 1.8,wvgrd < 1.95)))[0])
        for mask in telluric_mask:
            feats.remove('flux_'+str(mask))
            if difference==True:
                feats.remove('diff_'+str(mask))
    else:
        pass

    xlist = np.array(bin_df[feats]) #data

    typelist = ['primary_type','secondary_type']
    y=[]
    for i in range(len(bin_df)):
        zz = []
        for j in typelist:
            zz.append(bin_df[j][i])
        y.append(zz)
    ylist = np.array(y)

    if shape==True:
        print(xlist.shape, ylist.shape)

    # spit features and target variables into train and test split. Train set will have 70% of the features and the test will have 30% of the features.
    x_train, x_test, y_train, y_test = train_test_split(xlist, ylist, test_size=test_size, random_state=random_state, shuffle=shuffle)
    clf = MultiOutputRegressor(RandomForestRegressor(max_depth=max_depth, random_state=random_state,n_estimators=n_estimators,min_samples_leaf=min_samples_leaf,min_samples_split=min_samples_split,n_jobs=n_jobs))
    clf.fit(x_train, y_train) #fitting to the training set 

    datadic={}
    if (traindata==True):
        datadic['x_train']=x_train
        datadic['y_train']=y_train
    if (testdata==True):
        datadic['x_test']=x_test
        datadic['y_test']=y_test
    
    if (traindata==True)|(testdata==True):
        return clf, datadic
    else:
        return clf

def binaryPrecision(X_test,Y_test,model,predictions=False
                    # ,outprim1=1,outsec1=1,outprim3=3,outsec3=3
                    ):
    """
    Classifies the two stars that compose the given binaries by using the given multi output model and compares with their given types.
    Parameters
    ----------
    X_test : numpy array of numpy arrays
                    each array has to have 409 floats of the individual fluxes
    Y_test : numpy array of numpy arrays
                    each array has to have 409 floats with the real classifications
    
    model : sklearn.multioutput.MultiOutputRegressor
    predictions : bool
                default=False
    Returns
    -------
    df_avgdiffprim: pandas dataframe of the average difference between the predicted and actual type of the primaries for each group
    df_avgdiffseco: pandas dataframe of the average difference between the predicted and actual type of the secondaries for each group
    df_stdprim: pandas dataframe of the standard deviation between the predicted and actual type of the primaries for each group
    df_stdseco: pandas dataframe of the standard deviation between the predicted and actual type of the secondaries for each group
    Examples
    --------
    >>> df_avgdiffprim, df_avgdiffseco, df_stdprim, df_stdseco = binaryClassificationPrecision(flux_data,classification_data,model)
    """
    dic_b = {'primary_type': [i for i in list(range(16,40)) for j in range(16,40)],
         'secondary_type': [i for j in range(16,40) for i in list(range(16,40))]}
    types_df = pd.DataFrame(dic_b)
    types_df = types_df.loc[types_df['primary_type']<=types_df['secondary_type']].reset_index(drop=True)
    types_count = types_df.groupby('secondary_type').primary_type.value_counts().unstack()
    # df_avgdiffprim=types_count.copy()
    # df_avgdiffseco=types_count.copy()
    df_meddiffprim=types_count.copy()
    df_meddiffseco=types_count.copy()
    df_stdprim=types_count.copy()
    df_stdseco=types_count.copy()

    predsprim = []
    predssec = []
    realprim = []
    realsec = []

    for k1 in types_count.columns:
        med_diffprim_column = []
        med_diffseco_column = []
        std_diffprim_column = []
        std_diffseco_column = []

        for k2 in types_count.index:
            diffprim = []
            diffsec = []
            preds = []
            # predsprim=[]
            # predssec=[]
            # realprim=[]
            # realsec=[]
            for j in range(len(Y_test)):
                if (Y_test[j][0]==k1) & (Y_test[j][1]==k2):
                    realprim.append(Y_test[j][0])
                    realsec.append(Y_test[j][1])
                    preds.append(sorted(model.predict([X_test[j]])[0]))
                    
                    predsprim.append(preds[-1][0])
                    predssec.append(preds[-1][1])
                    # diffprim.append(predsprim[-1] - realprim[-1])
                    # diffsec.append(predssec[-1] - realsec[-1])
                    diffprim.append(preds[-1][0] - Y_test[j][0])
                    diffsec.append(preds[-1][1] - Y_test[j][1])
                    
            if len(diffprim) > 1:
                diffprim=np.array(diffprim)
                diffsec=np.array(diffsec)
                avg_diffprim=np.average(diffprim)
                med_diffprim_column.append(np.median(diffprim))

                avg_diffsec=np.average(diffsec)
                med_diffseco_column.append(np.median(diffsec))

                std_diffprim=np.sqrt(sum(np.abs(diffprim-avg_diffprim)**2)/(len(diffprim)-1))
                std_diffprim_column.append(std_diffprim)
                std_diffsec=np.sqrt(sum(np.abs(diffsec-avg_diffsec)**2)/(len(diffsec)-1))
                std_diffseco_column.append(std_diffsec)
            elif len(diffprim) == 1:
                diffprim=np.array(diffprim)
                diffsec=np.array(diffsec)
                avg_diffprim=np.average(diffprim)
                med_diffprim_column.append(np.median(diffprim))

                avg_diffsec=np.average(diffsec)
                med_diffseco_column.append(np.median(diffsec))

                std_diffprim=0
                std_diffprim_column.append(std_diffprim)
                std_diffsec=0
                std_diffseco_column.append(std_diffsec)  
            else:
                med_diffprim_column.append(np.nan)
                med_diffseco_column.append(np.nan)
                std_diffprim_column.append(np.nan)
                std_diffseco_column.append(np.nan)
        df_meddiffprim[k1]  = med_diffprim_column
        df_meddiffseco[k1]  = med_diffseco_column
        df_stdprim[k1]      = std_diffprim_column
        df_stdseco[k1]      = std_diffseco_column
    if predictions==True:
        return df_meddiffprim, df_meddiffseco, df_stdprim, df_stdseco, predsprim, predssec, realprim, realsec
    else:
        return df_meddiffprim, df_meddiffseco, df_stdprim, df_stdseco
    


def calculate_metrics(true_outputs, predicted_outputs, interval):
    '''
    '''
    unique_list = list(set(true_outputs))

    # Initialize the precision, f1-score, accuracy, and recall dictionaries
    tp = {}
    fn = {}
    fp = {}
    tn = {}
    negs = []
    precision = {}
    f1_score = {}
    accuracy = {}
    recall = {}

    for value in unique_list:
        value_position = np.where(true_outputs == value)[0]
        inlimit = np.where(np.abs(predicted_outputs-value)<=interval)[0]
        outlimit = np.where(np.abs(predicted_outputs-value)>interval)[0]

        tp[value] = sorted(list(set(value_position).intersection(inlimit)))
        negs     += sorted(list(set(value_position).intersection(outlimit)))
        fn[value] = sorted(list(set(value_position).intersection(outlimit)))



    for neg in negs:
        negpred = predicted_outputs[neg]
        closest_value = min(unique_list, key=lambda x: abs(x - negpred))
        try:
            fp[closest_value] += [neg]
        except KeyError:
            fp[closest_value] = [neg]

    for value in unique_list:
        if value not in fp.keys():
            fp[value]=[]
        used = tp[value]+fn[value]+fp[value]
        tn[value] = [i for i in range(len(true_outputs)) if i not in used]
        if len(tp[value])==0:
            precision[value] = 0
            recall[value] = 0
            f1_score[value] = 0
        else:
            precision[value] = len(tp[value])/(len(fp[value])+len(tp[value]))
            recall[value] = len(tp[value])/(len(fn[value])+len(tp[value]))
            f1_score[value] = 2*precision[value]*recall[value]/(precision[value] + recall[value])
        accuracy[value] = (len(tp[value]) + len(tn[value]))/(len(fp[value]) + len(tp[value]) + len(fn[value]) + len(tn[value]))

    fp = {k: fp[k] for k in sorted(fp)}

    return precision, recall, f1_score, accuracy


def unique(list1):
    # initialize a null list
    unique_list = []
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

def _RFclassify(DF_,wave,telluric=False,optimization=False,jobs=-1,Group=''):

    Bardalez = ['G1','Bardalez',[16,27]]
    Burgasser = ['G2','Burgasser',[24,32]]
    alltypes = ['G3','all','',[16,37]]
    latetypes = ['G4','late',[27,37]]
    if Group in Bardalez:
        typerange=Bardalez[-1]
    elif Group in Burgasser:
        typerange=Burgasser[-1]
    elif Group in alltypes:
        typerange=alltypes[-1]
    elif Group in latetypes:
        typerange=latetypes[-1]
    elif (type(Group)==list):
        if (min(Group)>=16)&(max(Group)<=37):
            typerange=Group

    df_types=[]
    for flux_i,noise_i in zip(DF_.system_interpolated_flux,DF_.system_interpolated_noise):
        df_types.append(typeToNum(fast_classify(wave,flux_i,noise_i)))
    df_types = np.array(df_types)
    DF_ = DF_[(df_types>=typerange[0]) & (df_types<=typerange[-1])]
    if len(typerange)>2:
        result_list = [any(items) for items in zip(*[df_types==type_i for type_i in typerange])]
        DF_ = DF_[result_list]


    RSEED = 42  

    feats = ['flux_' + str(k) for k in range(409)]

    
    X0 = np.array(DF_[feats]) #data
    y0 = np.array(DF_['num_class']) #labels 

    X_train, X_test, y_train, y_test = train_test_split(X0,y0,train_size=0.7, random_state=RSEED, shuffle=True)
    if telluric:
        tm = np.logical_or(np.logical_and(wave > 1.35,wave < 1.42), np.logical_and(wave > 1.8,wave < 1.95))
        X_train = np.array([X_train[i][~tm] for i in range(len(X_train))])
        X_test = np.array([X_test[i][~tm] for i in range(len(X_test))])

    data_test = {'X_test':X_test,
                'y_test':y_test}

    RF = RandomForestClassifier(random_state=42,n_estimators=50,max_depth=15,min_samples_split=2,min_samples_leaf=1)
    RF.fit(X_train, y_train)
    return RF, data_test

def RFclassify(DF_,wave,telluric=False,optimization=False,jobs=-1,Group=''):

    Bardalez = ['G1','Bardalez']
    Burgasser = ['G2','Burgasser']
    alltypes = ['G3','all','']
    earlytypes = ['G4','late']
    if Group in Bardalez:
        typerange=[[16,27],[31,38]]
    elif Group in Burgasser:
        typerange=[[25,32],[32,38]]
    elif Group in alltypes:
        typerange=[[16,39],[16,39]]
    elif Group in earlytypes:
        typerange=[[16,30],[16,32]]
    elif (type(Group)==list):
        if len(Group)==2:
            if (len(Group[0])==2)&(len(Group[1])==2):
                typerange=Group
    else:
        return print('Error. Select a valid group')
    
    
    sins=DF_[DF_.num_class==0]
    bins=DF_[DF_.num_class==1]
    
    if 'primary_type' in DF_.columns:
        bins = bins[bins.primary_type>=typerange[0][0]]
        bins = bins[bins.primary_type<=typerange[0][1]]
        bins = bins[bins.secondary_type>=typerange[1][0]]
        bins = bins[bins.secondary_type<=typerange[1][1]]
    else:
        return print('Include a column with the primary and secondary types for the binaries')

    if 'system_interpolated_flux' in DF_.columns:
        fluxcolname = 'system_interpolated_flux'
    elif 'FLUX' in DF_.columns:
        fluxcolname = 'FLUX'
    else: return print('Missing flux column')
    if 'system_interpolated_noise' in DF_.columns:
        noisecolname = 'system_interpolated_noise'
    elif 'UNCERTAINTY' in DF_.columns:
        noisecolname = 'UNCERTAINTY'
    else: return print('Missing noise column')

    bin_types=[]
    for flux_i,noise_i in zip(bins[fluxcolname],bins[noisecolname]):
        bin_types.append(typeToNum(fast_classify(wave,flux_i,noise_i)))

    sin_types=[]
    for flux_i,noise_i in zip(sins[fluxcolname],sins[noisecolname]):
        sin_types.append(typeToNum(fast_classify(wave,flux_i,noise_i)))
    sin_types = np.array(sin_types)
    sins = sins[(sin_types>=min(bin_types)) & (sin_types<=max(bin_types))]
    
    DF_ = pd.concat([sins,bins.drop(['primary_type','secondary_type'],axis=1)],axis=0)

    RSEED = 42  

    feats = ['flux_' + str(k) for k in range(409)]

    
    X0 = np.array(DF_[feats]) #data
    y0 = np.array(DF_['num_class']) #labels 

    X_train, X_test, y_train, y_test = train_test_split(X0,y0,train_size=0.7, random_state=RSEED, shuffle=True)
    if telluric:
        tm = np.logical_or(np.logical_and(wave > 1.35,wave < 1.42), np.logical_and(wave > 1.8,wave < 1.95))
        X_train = np.array([X_train[i][~tm] for i in range(len(X_train))])
        X_test = np.array([X_test[i][~tm] for i in range(len(X_test))])

    data_test = {'X_test':X_test,
                'y_test':y_test}

    RF = RandomForestClassifier(random_state=42,n_estimators=50,max_depth=15,min_samples_split=2,min_samples_leaf=1)
    RF.fit(X_train, y_train)
    return RF, data_test
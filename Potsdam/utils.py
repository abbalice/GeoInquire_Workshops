import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import argparse
from netCDF4 import Dataset
import os
import matplotlib
import warnings
warnings.filterwarnings("ignore")
import random
from numba import njit
import cartopy.crs as ccrs
import cartopy.feature as cfeature
# Set a Random seed for reproducibility
np.random.seed()
import time

rng = np.random.default_rng()

#----------------------------------------------------------------------------------------------
def extract_rates(filename):
    ids_scen     = []
    annual_rates = []
    f            = open(filename, "r")
    for iline, line in enumerate(f):
        tmp_id   = line.split(',')[0]
        tmpprob  = [float(prob) for prob in line.split(',')[1:]]
        ids_scen.append(tmp_id)
        annual_rates.append(tmpprob)
    ids_scen     = np.array(ids_scen)
    annual_rates = np.vstack((annual_rates))
    return ids_scen, annual_rates

#----------------------------------------------------------------------------------------------
def plot_hazard_curves_wmap_offshore(target, poi_lon, poi_lat, crop_region, thresholds, lambda_exc):

    fig, axes      = plt.subplots(
    figsize    = (13,5),
    nrows      = 1,
    ncols      = 2,
    subplot_kw = {'projection': ccrs.PlateCarree()}
    )

    """
    Curves
    """

    ax1 = axes[0]
    ax1.remove()
    ax1 = fig.add_subplot(1,2,1)

    for j in range(1000):
        ax1.plot(thresholds, lambda_exc[:, j], color = 'tab:blue', ls = '-', lw = 0.1, alpha = 0.2)

    # Mean
    lambda_exc_mean = lambda_exc.mean(axis = 1)

    # Percentiles of the distribution
    lambda_exc_p2  = np.percentile(lambda_exc, 2,  axis = 1)
    lambda_exc_p16 = np.percentile(lambda_exc, 16, axis = 1)
    lambda_exc_p84 = np.percentile(lambda_exc, 84, axis = 1)
    lambda_exc_p98 = np.percentile(lambda_exc, 98, axis = 1)

    ax1.plot(thresholds, lambda_exc_mean, color = 'red',        ls = '-', lw = 3,  label = 'Mean')
    ax1.plot(thresholds, lambda_exc_p2,   color = 'navy',       ls = '-', lw = 3,  label = '2-98th perc.')
    ax1.plot(thresholds, lambda_exc_p16,  color = 'tab:orange', ls = '-', lw = 3,  label = '16-84th perc.')
    ax1.plot(thresholds, lambda_exc_p84,  color = 'tab:orange', ls = '-', lw = 3,  label = '')
    ax1.plot(thresholds, lambda_exc_p98,  color = 'navy',       ls = '-', lw = 3,  label = '')

    #ax1.set_xscale('log')
    ax1.set_yscale('log')

    ax1.set_xlim([0, thresholds.max()])
    ax1.set_ylim([1e-6, 1])

    ax1.set_xlabel("Maximum wave height [m]", fontsize=13)
    ax1.set_ylabel("Exceedance-rate (1 yr)", fontsize=13)

    ax1.tick_params(labelsize=15)

    ax1.legend(fontsize = 12, frameon = False, loc = "upper right")
    ax1.set_title(f"Offshore POI {target}", fontsize = 16)

    """
    Map
    """

    ax2 = axes[1]

    ax2.set_extent(
        crop_region,
        crs=ccrs.PlateCarree()
    )

    ax2.add_feature(cfeature.OCEAN, facecolor = 'aliceblue')
    ax2.add_feature(cfeature.LAND,  facecolor = 'lightgray')

    ax2.scatter(
        poi_lon,
        poi_lat,
        s         = 90,
        color     = 'red',
        edgecolor = 'white',
        linewidth = 1.2,
        transform = ccrs.PlateCarree(),
        zorder    = 10
    )

    ax2.text(
        poi_lon - 0.05,
        poi_lat + 0.05,
        "POI",
        fontsize  = 12,
        color     = 'red',
        weight    = 'bold',
        transform = ccrs.PlateCarree()
    )

    gl = ax2.gridlines(
        draw_labels=True,
        linewidth=0.4,
        linestyle='--',
        alpha=0.5
    )

    gl.top_labels = False
    gl.right_labels = False
    plt.tight_layout()
    plt.show()
    return


#----------------------------------------------------------------------------------------------

def plot_hazard_curves_wmap_onshore(poi, lon0, lat0, thresholds, lambda_total, lon, lat, z):
    fig, axes      = plt.subplots(
        figsize    = (13,5),
        nrows      = 1,
        ncols      = 2,
        subplot_kw = {'projection': ccrs.PlateCarree()}
    )

    """
    Curves
    """

    ax1 = axes[0]
    ax1.remove()
    ax1 = fig.add_subplot(1,2,1)

    #for j in range(1000):
    #    ax1.plot(thresholds, lambda_exc[poi, :, j], color = 'tab:blue', ls = '-', lw = 0.1, alpha = 0.2)

    lambda_exc_mean = lambda_total.mean(axis = 1)
    lambda_exc_p2   = np.percentile(lambda_total, 2,   axis = 1)
    lambda_exc_p16  = np.percentile(lambda_total, 16,  axis = 1)
    lambda_exc_p84  = np.percentile(lambda_total, 84,  axis = 1)
    lambda_exc_p98  = np.percentile(lambda_total, 98,  axis = 1)

    ax1.plot(thresholds, lambda_exc_mean, color = 'red',        ls = '-', lw = 3,  label = 'Mean')
    ax1.plot(thresholds, lambda_exc_p2,   color = 'navy',       ls = '-', lw = 3,  label = '2-98th perc.')
    ax1.plot(thresholds, lambda_exc_p16,  color = 'tab:orange', ls = '-', lw = 3,  label = '16-84th perc.')
    ax1.plot(thresholds, lambda_exc_p84,  color = 'tab:orange', ls = '-', lw = 3,  label = '')
    ax1.plot(thresholds, lambda_exc_p98,  color = 'navy',       ls = '-', lw = 3,  label = '')


    #ax1.plot(thresholds, lambda_exc_mean[poi, :], color = 'red',        ls = '-', lw = 3,  label = 'Mean')
    #ax1.plot(thresholds, lambda_exc_p2[poi,:],   color = 'navy',       ls = '-', lw = 3,  label = '2-98th perc.')
    #ax1.plot(thresholds, lambda_exc_p16[poi,:],  color = 'tab:orange', ls = '-', lw = 3,  label = '16-84th perc.')
    #ax1.plot(thresholds, lambda_exc_p84[poi,:],  color = 'tab:orange', ls = '-', lw = 3,  label = '')
    #ax1.plot(thresholds, lambda_exc_p98[poi,:],  color = 'navy',       ls = '-', lw = 3,  label = '')


    #ax1.set_xscale('log')
    ax1.set_yscale('log')

    ax1.set_xlim([0, thresholds.max()])
    ax1.set_ylim([1e-6, 1])

    ax1.set_xlabel("Flowdepth [m]", fontsize=13)
    ax1.set_ylabel("Exceedance-rate (1 yr)", fontsize=13)

    ax1.tick_params(labelsize=15)

    ax1.legend(fontsize = 12, frameon = False, loc = "upper right")
    ax1.set_title(f"Point {poi+1}", fontsize = 16)

    """
    Map
    """
    z_bathy   = np.ma.masked_where(z > 0, z)
    lon_west  = lon.min()
    lon_east  = lon.max()
    lat_south = lat.min()
    lat_north = lat.max()

    ax2 = axes[1]

    ax2.set_extent([lon_west, lon_east, lat_south, lat_north], crs = ccrs.PlateCarree())

    # topography
    ax2.contourf(lon, lat, z, levels = 40, cmap = "gray", extend = "min", transform = ccrs.PlateCarree())

    # bathymetry
    ax2.contourf(lon, lat, z_bathy, levels = np.arange(-30, 1, 1), cmap = "Blues_r", extend = "min", transform=ccrs.PlateCarree())

    ax2.scatter(
        lon0[poi],
        lat0[poi],
        s         = 100,
        color     = "red",
        edgecolor = "white",
        linewidth = 1,
        transform = ccrs.PlateCarree(),
        zorder    = 10
    )

    ax2.text(
        lon0[poi] - 0.001,
        lat0[poi] + 0.001,
        f"Point {poi+1}",
        fontsize  = 12,
        color     = 'red',
        weight    = 'bold',
        transform = ccrs.PlateCarree()
    )

    gl = ax2.gridlines(draw_labels = True, alpha = 0.05)
    gl.top_labels   = False
    gl.right_labels = False


    plt.show()
    return

#----------------------------------------------------------------------------------------------

def plot_poe_dt(thresholds, lambda_exc):

    lambda_exc_mean  = lambda_exc.mean(axis=1)
    lambda_exc_p2    = np.percentile(lambda_exc,  2, axis = 1)
    lambda_exc_p16   = np.percentile(lambda_exc, 16, axis = 1)
    lambda_exc_p84   = np.percentile(lambda_exc, 84, axis = 1)
    lambda_exc_p98   = np.percentile(lambda_exc, 98, axis = 1)

    fig, ax1 = plt.subplots(figsize=(5,3))

    ax1.plot(thresholds, lambda_exc_mean, color = 'red',        ls = '-', lw = 3,  label = 'Mean')
    ax1.plot(thresholds, lambda_exc_p2,   color = 'navy',       ls = '-', lw = 3,  label = '2-98th perc.')
    ax1.plot(thresholds, lambda_exc_p16,  color = 'tab:orange', ls = '-', lw = 3,  label = '16-84th perc.')
    ax1.plot(thresholds, lambda_exc_p84,  color = 'tab:orange', ls = '-', lw = 3,  label = '')
    ax1.plot(thresholds, lambda_exc_p98,  color = 'navy',       ls = '-', lw = 3,  label = '')

    #ax1.set_xscale('log')
    ax1.set_yscale('log')

    ax1.set_xlim([0, thresholds.max()])
    ax1.set_ylim([5e-5, 1])

    ax1.set_xlabel("Flowdepth [m]", fontsize=10)
    ax1.set_ylabel("Probability of exceedance (50 yrs)", fontsize=10)

    ax1.tick_params(labelsize=10)
    ax1.legend(fontsize = 10, frameon = False, loc = "upper right")
    plt.show()
    return


#----------------------------------------------------------------------------------------------
def plot_intensity_for_poe(poi, lon0, lat0, lon, lat, z, thresholds, lambda_total, fd_hazard_map, pth, curve):

    fig, axes      = plt.subplots(
        figsize    = (13,5),
        nrows      = 1,
        ncols      = 2,
        subplot_kw = {'projection': ccrs.PlateCarree()}
        )

    """
    Curves
    """
    lambda_exc_mean = lambda_total.mean(axis = 1)
    lambda_exc_p2   = np.percentile(lambda_total, 2,   axis = 1)
    lambda_exc_p16  = np.percentile(lambda_total, 16,  axis = 1)
    lambda_exc_p84  = np.percentile(lambda_total, 84,  axis = 1)
    lambda_exc_p98  = np.percentile(lambda_total, 98,  axis = 1)

    ax1 = axes[0]
    ax1.remove()
    ax1 = fig.add_subplot(1,2,1)

    #-------

    ax1.plot(thresholds, lambda_exc_mean, color = 'lightgray',        ls = '-', lw = 3,  label = '')
    ax1.plot(thresholds, lambda_exc_p2,   color = 'lightgray',       ls = '-', lw = 3,  label = '')
    ax1.plot(thresholds, lambda_exc_p16,  color = 'lightgray', ls = '-', lw = 3,  label = '')
    ax1.plot(thresholds, lambda_exc_p84,  color = 'lightgray', ls = '-', lw = 3,  label = '')
    ax1.plot(thresholds, lambda_exc_p98,  color = 'lightgray',       ls = '-', lw = 3,  label = '')
    #--------
    if curve == 'mean':
        ax1.plot(thresholds, lambda_exc_mean, color = 'red',        ls = '-', lw = 3,  label = 'Mean')
    elif curve == 'p2':
        ax1.plot(thresholds, lambda_exc_p2,   color = 'navy',       ls = '-', lw = 3,  label = '2th perc.')
    elif curve == 'p16':
        ax1.plot(thresholds, lambda_exc_p16,  color = 'tab:orange', ls = '-', lw = 3,  label = '16th perc.')
    elif curve == 'p84':
        ax1.plot(thresholds, lambda_exc_p84,  color = 'tab:orange', ls = '-', lw = 3,  label = '84th perc.')
    elif curve == 'p98':
        ax1.plot(thresholds, lambda_exc_p98,  color = 'navy',       ls = '-', lw = 3,  label = '98th perc.')

    ax1.axhline(pth, color='k', ls='-.', lw=2)
    ax1.text(x=5, y=pth + 0.05, s=f'Prob. {pth*100} %')
    ax1.axvline(fd_hazard_map, color='k', ls='-.', lw=2)

    #ax1.set_xscale('log')
    ax1.set_yscale('log')

    ax1.set_xlim([0, thresholds.max()])
    ax1.set_ylim([5e-5, 1])

    ax1.set_xlabel("Flowdepth [m]", fontsize=13)
    ax1.set_ylabel("Probability of exceedance (50 yrs)", fontsize=13)

    ax1.tick_params(labelsize=15)

    ax1.legend(fontsize = 12, frameon = False, loc = "upper right")
    ax1.set_title(f"Point {poi+1}", fontsize = 16)

    """
    Map
    """
    z_bathy   = np.ma.masked_where(z > 0, z)
    lon_west  = lon.min()
    lon_east  = lon.max()
    lat_south = lat.min()
    lat_north = lat.max()

    ax2 = axes[1]

    ax2.set_extent([lon_west, lon_east, lat_south, lat_north], crs=ccrs.PlateCarree())

    # topography
    cf = ax2.contourf(
        lon,
        lat,
        z,
        levels = 40,
        cmap   = "gray",
        extend = "min",
        transform=ccrs.PlateCarree()
    )

    # bathymetry
    cf = ax2.contourf(
        lon,
        lat,
        z_bathy,
        levels = np.arange(-30, 1, 1),
        cmap   = "Blues_r",
        extend = "min",
        transform=ccrs.PlateCarree()
    )


    # point
    ax2.scatter(lon0[poi], lat0[poi],
        s         = 100*fd_hazard_map, #100,
        color     = "red",
        edgecolor = "white",
        linewidth = 1,
        transform = ccrs.PlateCarree(),
        zorder    = 10
    )

    ax2.text(
        lon0[poi] - 0.001,
        lat0[poi] + 0.001,
        f"Fd = {round(fd_hazard_map, 2)} m",
        fontsize  = 12,
        color     = 'red',
        weight    = 'bold',
        transform = ccrs.PlateCarree()
    )


    gl = ax2.gridlines(draw_labels = True, alpha = 0.05)

    gl.top_labels   = False
    gl.right_labels = False
    plt.show()
    return
~                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
~                                                                                                                                

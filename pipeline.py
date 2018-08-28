#
#
# Copyright (C) 2017 Francisco Munoz, Patricio Rojo
#
from __future__ import print_function

import dataproc as dp
import dataproc.timeseries as tm
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import pdb
import glob
import logging
#import pyfits as pf
from astropy.io import fits as pf
import re

default_logger = logging.getLogger('pipeline')
handler_console = logging.StreamHandler()
formatter_console = logging.Formatter('%(message)s')
handler_console.setFormatter(formatter_console)
handler_console.setLevel(30)
default_logger.addHandler(handler_console)

# pipeline incomplete, for some cases could not work.

# Dictionaries for each telescope

dk154_2017 = dict(epoch="JD", sort_by="JD", gain = 0.24, ron = 4.7,
                  flat_in_hdr="FLAT", bias_in_hdr="BIAS", filter_in_hdr="FILTB",
                  imtype_in_hdr="IMAGETYP", telescope="danish", fmt = "fits",
                  auto_trim="TRIMSEC")

dk154_old = dict(epoch="JD", sort_by="JD", gain = 1.31, ron = 7.2,
                  flat_in_hdr="FLAT", bias_in_hdr="BIAS", filter_in_hdr="FILTB",
                  imtype_in_hdr="IMAGETYP", telescope="danish", fmt = "fits",
                  auto_trim="TRIMSEC")


warsaw_2013 = dict(epoch="JULDAT", sort_by="JULDAT", gain=1.8, ron=5.1,
              flat_in_hdr="illum", bias_in_hdr="zero", filter_in_hdr="FILTER",
              imtype_in_hdr="IMAGETYP", telescope="warsaw", fmt="fts",
              auto_trim=None)

warsaw_2016 = dict(epoch="JULDAT", sort_by="JULDAT", gain=1.6, ron=4.8,
              flat_in_hdr="flat", bias_in_hdr="zero", filter_in_hdr="FILTER",
              imtype_in_hdr="IMAGETYP", telescope="warsaw", fmt="ftsc",
              auto_trim=None)

# Gain and Read out noise depend of the quadrant / TODO: Recheck gain & ron values for each dict
ctio_1m = dict(epoch="JD", sort_by="JD", gain=1.33, ron=4.78,
               flat_in_hdr="FLAT", bias_in_hdr="BIAS", filter_in_hdr="FILTERID",
               imtype_in_hdr="IMGTYPE", telescope="ctio_1m", fmt="fits",
               auto_trim=None)

ctio_1m_dark = dict(epoch="JD", sort_by="JD", gain=1.33, ron=4.78,
               flat_in_hdr="FLAT", bias_in_hdr="Dark", filter_in_hdr="FILTERID",
               imtype_in_hdr="IMGTYPE", telescope="ctio_1m", fmt="fits",
               auto_trim=None)

ctio_1m_dome_2008 = dict(epoch="JD", sort_by="JD", gain=1.33, ron=4.78, flat_in_hdr="FLAT",
                    bias_in_hdr="ZERO", filter_in_hdr="FILTERID", imtype_in_hdr="IMGTYPE",
      	            telescope="ctio_1m", fmt="fits", auto_trim=None)

ctio_09m = dict(epoch="JD", sort_by="JD", gain=3, ron=12, flat_in_hdr="sflat",
                bias_in_hdr="zero", filter_in_hdr="FILTER2", imtype_in_hdr="IMAGETYP",
                telescope="ctio_09m", fmt="fits", auto_trim=None)

ctio_09m_dome = dict(epoch="JD", sort_by="JD", gain=3, ron=12, flat_in_hdr="dflat",
                     bias_in_hdr="zero", filter_in_hdr="FILTER2", imtype_in_hdr="IMAGETYP",
                     telescope="ctio_09m", fmt="fits", auto_trim=None)   

ctio_09m_dome_2013 = dict(epoch="JD", sort_by="JD", gain=3, ron=12, flat_in_hdr="DOME FLAT",
                          bias_in_hdr="BIAS", filter_in_hdr="FILTER2", imtype_in_hdr="IMAGETYP",
                          telescope="ctio_09m", fmt="fits", auto_trim=None)

rem_2011 = dict(epoch="JD", sort_by="JD", gain=2, ron=14,
                flat_in_hdr="FLATF", bias_in_hdr="DARKF", filter_in_hdr="FILTER",
                imtype_in_hdr="OBSTYPE", fmt="fits", auto_trim=None, telescope="rem_2011")

dome_soar = dict(epoch="JD",sort_by="JD",gain=1.4,ron=4.74,flat_in_hdr="DFLAT",
                bias_in_hdr="ZERO", filter_in_hdr="FILTER2", imtype_in_hdr="OBSTYPE", fmt="fits",
                telescope="soar",auto_trim=None)

swope = dict(epoch="JD", sort_by="JD", gain=1.040, ron=3.4, flat_in_hdr="Flat",
             bias_in_hdr="Bias", filter_in_hdr="WHEEL2", imtype_in_hdr="EXPTYPE",
             telescope="swope", fmt="fits", auto_trim=None)


def pipeline(target, files_path="raw/*", calib_path=None, object_in_hdr=None, coords_xy=None,
             stamp_rad=None, max_counts=500000, epoch='JULDAT',
             recenter=False, brightest=None, deg=-1, ccd_lims_xy=None,
             aps=None, sky_test=None, sector=None, sort_by='JULDAT',
             labelsize=14, interactive=False, offsets_xy=None,
             first_frame=None, last_frame=None, gain=None, ron=None, filter_band="I",
             flat_in_hdr=None, bias_in_hdr=None, filter_in_hdr=None, imtype_in_hdr=None,
             telescope=None, auto_trim=None, fmt="fits", hdu=None, hdud=None, exposure=None):

    """
    reduce datas and generate figures, a pdf with them and several data files
    with the important information.
    :param files_path: directory where the original files are saved
    :param calib_path: directory where the calibration files are saved
    :param object_in_hdr: name of the planet as is written in the header
    :param filter_band: filter on which the observations where taken
    :param target: name of the objetive star with the exoplanet
    :param coords_xy: coords of target and references star in the form [[x1,y1], [x2,y2], ...]
    :param labels: names used for target and references.
    :param stamp_rad: radius of the stamp used, by default, set at the value of 35 pixels
    :param max_counts: max number of counts allow in order to avoid saturated pixels, by default use 500000
    :param epoch: specify the epoch used, by default use Julian Date.
    :param recenter: if it is true, does a process of recentering the coordinates.
    :param brightest: specify the brightest star
    :param deg:
    :param ccd_lims_xy: specify limits for the CCD pixels, by default use the full CCD
    :param aps: range of aperture, in the form [ap_min, ap_max]
    :param sky_test: test values of sky
    :param sector: portion of the time series to use to find the optimal values of aperture and sky
    :param sort_by: parameter used to sort the datas, use JULDAT by default.
    :param labelsize: size of the plot's labels and titles.
    :param interactive: interactive option from dataproc, allow to check offsets on the timeseries.
    :param offsets_xy: offsets produced during the time series, in the form {frame_numb: [off_x, off_y], ...}.
    :param first_frame: first frame of the time series.
    :param last_frame: last frame of the time series.
    :param gain: gain of the telescope, for warsaw the pipeline extract this value from header.
    :param ron: readout noise of the telescope, for warsaw the pipeline extract this value from header.
    :return: the timeseries and photometry object generated from dataproc,
    """
    # set size for labels, axis ticks and titles
    if target is None:
        raise TypeError("user must specify a TARGET name")

    if object_in_hdr is None:
        object_in_hdr = target

    labels = [target] + ["ref{}".format(i+1) for i in range(len(coords_xy)-1)]


    matplotlib.rc('xtick', labelsize=labelsize)
    matplotlib.rc('ytick', labelsize=labelsize)
    matplotlib.rc('axes', titlesize=labelsize)
    matplotlib.rc('axes', labelsize=labelsize+1)

    if hdu is None:
        if fmt == "ftsc":
            hdu = 1
        else:
            hdu = 0
        hdud = hdu

    elif telescope == "soar":
        hdud = 5
        merge(files_path,calib_path,hdud=hdud)

    if calib_path is None:
        calib_path = files_path

    # find the format of the images and generate the astrodir,
    # master bias and the master flat.

    #fmt = find_format(files_path)
    
    mbias = master_bias(calib_path, fmt=fmt, bias_in_hdr=bias_in_hdr, imtype_in_hdr=imtype_in_hdr, hdu=hdu, hdud=hdud, exposure=exposure)
    files = astrodir_files(files_path, object_in_hdr, sort_by=sort_by, fmt=fmt,
                           first_frame=first_frame, last_frame=last_frame, auto_trim=auto_trim,
                           telescope=telescope, filter_band=filter_band, hdu=hdu, hdud=hdud)
    #fmt_calib = find_format(calib_path)
    mflat_normalized = norm_mflat(calib_path, fmt=fmt, mbias=mbias, filter_band=filter_band,
                                  flat_in_hdr=flat_in_hdr, bias_in_hdr=bias_in_hdr, filter_in_hdr=filter_in_hdr,
                                  auto_trim=auto_trim, imtype_in_hdr=imtype_in_hdr, hdu=hdu, hdud=hdud)
    if stamp_rad is None:
        default_logger.warning("Using default stamp radius {}".format(35))
        stamp_rad = 35

    if gain is None:
        #gain = files[0].getheaderval("gain")[0]
        gain = files[0].getheaderval("GAIN")[0]

    if ron is None:
        ron = files[0].getheaderval("RDNOISE")[0]

    # aplicate photometry
    phot = tm.Photometry(files, mdark=mbias, mflat=mflat_normalized,
                         target_coords_xy=coords_xy, labels=labels,
                         gain=gain, ron=ron, stamp_rad=stamp_rad,
                         max_counts=max_counts, epoch=epoch, recenter=recenter,
                         brightest=brightest, deg=deg, ccd_lims_xy=ccd_lims_xy,
                         interactive=interactive, offsets_xy=offsets_xy)

    # set range of apertures by default
    if aps is None:
        default_logger.warning("Using default range of apertures [{},{}]".format(10, 20))
        aps = [10, 20]
    # set sky test by default
    if sky_test is None:
        sky_test = [aps[1] + 1, stamp_rad - 1]
        default_logger.warning("Using default sky test of [{},{}] "
                               "pixels".format(sky_test[0], sky_test[1]))

    # set default aperture and sky (not optimal)
    ap = int(aps[1] / 2.0 + aps[0] / 2.0)
    sky_in = sky_test[0]
    sky_out = sky_test[1]

    # if sector is none it will use default aperture and sky
    if sector is None:
        default_logger.warning("Iteration not specified, using default aperture "
                               "({}) and sky ([{},{}])".format(ap, sky_in, sky_out))

    # if sector is not none, then find the optimal apertures in this sector
    # (out of transit datas) of the time series and the optimal sky
    if sector is not None:
        ap = find_ap(phot, aps, sky_test=sky_test, sector=sector, labelsize=labelsize)
        sky_in, sky_out = get_std_areas(phot, ap, sector=sector, stamp_rad=stamp_rad,
                                        labelsize=labelsize)
        sky_test = [sky_in, sky_out]
        sky_in, sky_out = find_sky(phot, ap, stamp_rad, sky_range=None,
                                   sky_test=sky_test, sector=sector,
                                   labelsize=labelsize)
        plt.close('all')

    # generate the time series of the target and references for aperture and sky optimal
    ts = phot.photometry(aperture=ap, sky=[sky_in, sky_out])
    
    matplotlib.rc('axes', titlesize=labelsize)

    # plot the exptime per frame
    exptime(files, save='figures/exptime.png', labelsize=labelsize, telescope=telescope)

    # plot flux and flux normalized
    plot_flux(ts, labels, ap, normalize=False, save="figures/flux.png",
              axes=20, labelsize=labelsize)
    plot_flux(ts, labels, ap, normalize=True, save="figures/normflux.png",
              axes=21, labelsize=labelsize)
    # plot ratio
    fig = plt.figure()
    ts.plot_ratio(label=labels[0], save="figures/ratio.png", axes=fig, overwrite=True)
    ax = fig.add_subplot(111)
    ax.set_title("Flux ratio of {} (ap={},sky=[{},{}])".format(labels[0], ap, sky_in, sky_out))
    if sector is not None:
        ax.axvspan(phot.epoch[sector[0]], phot.epoch[sector[1]], facecolor='#2ca02c', alpha=0.5)
    plt.tight_layout()
    plt.savefig("figures/ratio.png")

    # plot radial profile
    fig = plt.figure()

    phot.plot_radialprofile(targets=target, recenter=True, frame=0,
                            save="figures/radialprof.png", axes=fig, overwrite=True)

    ax = fig.add_subplot(111)
    ax.axvline(x=ap)
    ax.axvspan(aps[0], aps[1], facecolor='#2ca02c', alpha=0.5)
    plt.tight_layout()
    plt.savefig("figures/radialprof.png")

    # plot reference CCD, first frame with the target and references pointed out.
    fig = plt.figure()
    phot.imshowz(axes=fig, save="figures/ref.png", overwrite=True,
                 ccd_lims_xy=ccd_lims_xy, mflat=mflat_normalized)
    ax = fig.add_subplot(111)
    ax.set_title("Primary and references")
    ax.locator_params(nbins=3, axis='x')
    plt.tight_layout()
    plt.savefig("figures/ref.png")

    # plot stamps for target of every frame.
    fig = plt.figure()
    phot.showstamp(target=labels[0], axes=fig, save="figures/stamp.png", overwrite=True)
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.set_title("Primary target, stamps radius: " + str(stamp_rad))
    plt.tight_layout()
    plt.savefig("figures/stamp.png")

    # if sector is not none, then it will create data files and the PDF extraction.
    if sector is not None:

        # check if figures and data directories exists or created them.
        check_dirs("figures")
        check_dirs("data")

        # extracta flux data (with errors) for target and references
        flux_extraction(phot, ts, labels, ap)

        # extract rest of the data for target and references
        zip_data(ts, phot, ap, labels)

        # add the extra plots (related to the rest of the data)
        plot_extra_data(phot, ts, ap, labels, force_show=False, labelsize=labelsize)

        # check if the file tramos_custom.sty exists or create it
        make_tramos_custom(ap, target)

        # generate pdf file with the plots
        generate_extraction(labels[0])

    #  make sure of close all the figures that was created.
    plt.close('all')
    return ts, phot


def plot_extra_data(phot, ts, ap, labels, force_show=True, labelsize=14):
    """
    generate plots for extra data from time series (like momentums or FWHM).
    :param phot: photometric object from dataproc
    :param ts: timeseries object from dataproc
    :param ap: photometric aperture
    :param labels: label names for target and references
    :param force_show: if it is true, then show the plot
    :param labelsize: size of the plot's labels and titles.
    :return:
    """
    matplotlib.rc('axes', titlesize=labelsize + 2)
    matplotlib.rc('axes', labelsize=labelsize + 1)
    matplotlib.rc('xtick', labelsize=labelsize + 1)
    matplotlib.rc('ytick', labelsize=labelsize + 1)
    matplotlib.rc('legend', fontsize=labelsize)

    infos = ["mom3_mag_ap" + str(ap),
             "mom3_ang_ap" + str(ap), "fwhm",
             "excess_ap" + str(ap), "peak_ap" + str(ap)]
    y_labels = [r"3rd mom(mag) [$pixel^3$]",
                "3rd mom(ang)", "Fwhm", "Excess", "Peak"]

    titles = ["3rd mom(mag)", "3rd mom(ang)",
              "Fwhm", "Excess", "Peak"]
    save = ["mom3_mag.png", "mom3_ang.png",
            "fwhm.png", "excess.png", "peak.png"]
    i = 0
    for inf in infos:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for lab in labels:
            ax.plot(phot.epoch, ts(inf)[lab], '-', label=lab)
        ax.set_title(titles[i] + " for target and refs")
        ax.set_xlabel("Epoch (JD)")
        ax.set_ylabel(y_labels[i])
        lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if force_show:
            plt.show()
        else:
            plt.savefig("figures/" + save[i], bbox_extra_artists=(lgd,),
                        bbox_inches="tight")
        i += 1

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for lab in labels:
        ax.plot(phot.epoch, ts("mom2_mag_ap" + str(ap))[lab], '-', label=lab)
    ratio, ratio_error, sigma, errb_m = ts.get_ratio()
    ax.set_title("2nd mom(mag)")
    ax.set_xlabel("Epoch (JD)")
    ax.set_ylabel(r"2nd mom(mag) [$pixel^2$]")
    ax2 = ax.twinx()
    ax2.fill_between(phot.epoch, ratio, 0, facecolor='blue', alpha=0.35)
    ax2.set_ylabel('ratio', color='b')
    ax2.tick_params('y', colors='b')
    ax2.set_ylim([ratio.min() - ratio_error.max() - 0.1,
                  ratio.max() + ratio_error.max() + 0.1])
    plt.tight_layout()
    lgd = ax.legend(loc='center left', bbox_to_anchor=(1.15, 0.85))
    if force_show:
        plt.show()
    else:
        plt.savefig("figures/mom2_mag.png", bbox_extra_artists=(lgd,),
                    bbox_inches="tight")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    var_x = (ts("centers_xy")[labels[0]][:, 0] - ts("centers_xy")[labels[0]][0, 0])
    var_y = (ts("centers_xy")[labels[0]][:, 1] - ts("centers_xy")[labels[0]][0, 1])
    ax.plot(var_x, var_y, '-')
    ax.set_title("Variations of centers (x,y)")
    ax.set_xlabel(r"$\Delta x$")
    ax.set_ylabel(r"$\Delta y$")
    plt.tight_layout()
    if force_show:
        plt.show()
    else:
        plt.savefig("figures/centers_xy.png")


def zip_data(ts, phot, ap, labels):
    """
    generate data files of detail data for target and references
    :param ts: timeseries object from dataproc
    :param phot: photometric object from dataproc
    :param ap: photometric aperture
    :param labels: label names for target and references
    :return:
    """
    infos = ["mom2_mag_ap" + str(ap), "mom3_mag_ap" + str(ap),
             "mom3_ang_ap" + str(ap), "fwhm",
             "excess_ap" + str(ap), "peak_ap" + str(ap)]
    header = "epoch(JD) " + ' '.join(infos) + " centers_x centers_y"
    for lab in labels:
        centers = zip(*[ts("centers_xy")[lab][:, x] for x in range(2)])
        other_datas = zip(phot.epoch, *[ts(x)[lab] for x in infos])
        data_detail = [d1 + d2 for d1, d2 in zip(other_datas, centers)]
        save_detail = "data/ts_detail_" + lab + ".dat"
        sp.savetxt(save_detail, data_detail, header=header, delimiter=" ", fmt="%s")


def flux_extraction(phot, ts, labels, ap):
    """
    generate data file of the flux (with error) obtained for target and references
    :param phot: photometric object from dataproc
    :param ts: timeseries object from dataproc
    :param labels: label names for target and references
    :param ap: photometric aperture
    :return:
    """
    header_array = sp.array("epoch(JD)")
    epoch_lab = sp.array(phot.epoch)
    flux_0 = ts('flux_ap' + str(ap))[labels[0]]
    error_0 = ts('flux_ap' + str(ap)).errors[0]
    dat = sp.column_stack((epoch_lab, flux_0))
    dat = sp.column_stack((dat, error_0))
    for i in range(len(labels)-1):
        lab = labels[i+1]
        flux_lab = ts('flux_ap' + str(ap))[lab]
        error_lab = ts('flux_ap' + str(ap)).errors[i+1]
        header_array = sp.append(header_array, lab)
        lab_error = "error_" + lab
        header_array = sp.append(header_array, lab_error)
        dat_dummy = sp.column_stack((flux_lab, error_lab))
        dat = sp.column_stack((dat, dat_dummy))
    save_txt = "data/ts_fluxes.dat"
    header = ' '.join(header_array)
    sp.savetxt(save_txt, dat, header=header, delimiter=" ", fmt="%s")


def plot_flux(ts, labels, ap,  normalize=False, save=None, axes=None, labelsize=14):
    """
    plot of flux with errorbar for target and references
    :param ts: timeseries object from dataproc
    :param labels: label names for target and references
    :param ap: photometric aperture
    :param normalize: if it is true, then normalize the flux by mean value
    :param save: name of the plot with directory, example 'directory/plot.png'
    :param axes:
    :param labelsize:
    :return:
    """
    matplotlib.rc('axes', titlesize=labelsize)
    fig = plt.figure(axes)
    for lab in labels:
        ts.plot(label=lab, axes=fig, normalize=normalize, save=save, overwrite=True)

    ax = fig.add_subplot(111)
    ax.set_title("Flux at aperture " + str(ap))
    lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    if save is not None:
        plt.savefig(save, bbox_extra_artists=(lgd,), bbox_inches="tight")
    else:
        plt.show()


def get_min_std_ap(phot, aps, sky_test, sector=None, save=None):
    """
    iterate a given range of apertures, obtaining different values of ratio
    out of transit, then, in every cycle, evaluate the standard deviation of the
    corresponding time series out of transit and find the minimum deviation.
    :param phot: photometric object from dataproc
    :param aps: range of photometric apertures
    :param sky_test: test values of photometric sky
    :param sector: sector of the time series out of transit
    :param save: name to save the plot.
    :return: standard deviation, optimal aperture and mean errorbar
    """
    desv = sp.array([])
    errb = sp.array([])
    if isinstance(aps, sp.ndarray):
        fig = plt.figure()
        for aperture in aps:
            ts = phot.photometry(aperture=aperture, sky=sky_test)
            ratio, ratio_error, sigma, errb_m = ts.get_ratio(sector=sector)
            label = ','.join(["apert" + str(aperture),
                              r" $\sigma$" + str(round(sigma, 3)),
                              r" $errorbar$" + str(round(errb_m, 3))])
            ts.plot_ratio(label=label, axes=fig, sector=sector, overwrite=True, save=save)
            desv = sp.append(desv, sigma)
            errb = sp.append(errb, errb_m)
        ax = fig.add_subplot(111)
        ax.locator_params(nbins=7, axis='x')
        lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if save is not None:
            plt.savefig(save, bbox_extra_artists=(lgd,), bbox_inches="tight")
        else:
            plt.show()
        opt_ap = aps[sp.argmin(desv)]

        return desv, opt_ap, errb


def get_opt_sky_in(phot, ap, in_array, sky_test=None, sector=None, save=None,
                   labelsize=14):
    """
    iterate a given range of sky inner radius, obtaining different values of ratio
    out of transit, then, in every cycle, evaluate the standard deviation of the
    corresponding time series out of transit and find the minimum deviation.
    :param phot: photometric object from dataproc
    :param ap: aperture value for the timeseries.
    :param in_array: array of inner skies radius.
    :param sky_test: test values of photometric sky
    :param sector: sector of the time series out of transit.
    :param save: name to save the plot.
    :param labelsize:
    :return: standard deviation, optimal inner sky radius and mean errorbar
    """
    desv = sp.array([])
    errbar = sp.array([])
    fig = plt.figure()
    area_0 = areas(ap, in_array[0], sky_test[1])
    for sky_in in in_array:
        area = areas(ap, sky_in, sky_test[1])
        razon = round(area / area_0, 2)
        ts = phot.photometry(aperture=ap, sky=[sky_in, sky_test[1]])
        ratio, ratio_error, sigma, errb_m = ts.get_ratio(sector=sector)
        label = "sky [" + str(sky_in) + "," + str(sky_test[1]) + r"]: $area/area_0$" + str(razon)
        ts.plot_ratio(label=label, axes=fig, sector=sector, overwrite=True, save=save)
        desv = sp.append(desv, sigma)
        errbar = sp.append(errbar, errb_m)
    ax = fig.add_subplot(111)
    ax.locator_params(nbins=7, axis='x')
    ax.set_title("Ratio at aperture {}, several sky annulus".format(ap))
    lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(save, bbox_extra_artists=(lgd,), bbox_inches="tight")
    opt_sky_in = in_array[sp.argmin(desv)]
    return desv, opt_sky_in, errbar


def get_opt_sky_out(phot, ap, sky_in, out_array, sector=None, save=None):
    """
    iterate a given range of sky external radius, obtaining different values of ratio
    out of transit, then, in every cycle, evaluate the standard deviation of the
    corresponding time series out of transit and find the minimum deviation.
    :param phot: photometric object from dataproc
    :param ap: aperture value for the timeseries.
    :param sky_in: inner sky radius value for the timeseries.
    :param out_array: array of external skies radius.
    :param sector: sector of the time series out of transit.
    :param save: name to save the plot.
    :return: standard deviation, optimal external sky radius and mean errorbar
    """
    desv = sp.array([])
    errbar = sp.array([])
    fig2 = plt.figure()
    area_0 = areas(ap, sky_in, out_array[0])
    for sky_out in out_array:
        area = areas(ap, sky_in, sky_out)
        razon = round(area / area_0, 2)
        ts = phot.photometry(aperture=ap, sky=[sky_in, sky_out])
        ratio, ratio_error, sigma, errb_m = ts.get_ratio(sector=sector)
        label = "sky [" + str(sky_in) + "," + str(sky_out) + r"]: $area/area_0$" + str(razon)
        ts.plot_ratio(label=label, axes=fig2, sector=sector, overwrite=True, save=save)
        desv = sp.append(desv, sigma)
        errbar = sp.append(errbar, errb_m)
    ax2 = fig2.add_subplot(111)
    ax2.locator_params(nbins=7, axis='x')
    ax2.set_title("Ratio at aperture {}, several sky annulus".format(ap))
    lgd = ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(save, bbox_extra_artists=(lgd,), bbox_inches="tight")
    opt_sky_out = out_array[sp.argmin(desv)]
    return desv, opt_sky_out, errbar


def get_std_areas(phot, ap, sector=None, stamp_rad=None, labelsize=14):
    """
    iterate a given range of skys [in, out], keeping the relation of areas
    constant (out**2 - in**2 = cte) obtaining different values of ratio
    out of transit, then, in every cycle, evaluate the standard deviation of the
    corresponding time series out of transit and find the minimum deviation.
    :param phot: photometric object from dataproc
    :param ap: aperture value for the timeseries.
    :param sector: sector of the time series out of transit.
    :param stamp_rad: radius of the stamp
    :param labelsize:
    :return: inner and external sky radius of the minimum standard deviation.
    """
    in_array = sp.array([ap])
    out_array = sp.array([ap + 8])
    area_0 = areas(ap, in_array[0], out_array[0])
    ratio_area = sp.array([])
    desv = sp.array([])
    errbar = sp.array([])
    fig = plt.figure()
    i = 0
    while out_array[i] < stamp_rad:
        area = areas(ap, in_array[i], out_array[i])
        ratio_area_i = round(area / area_0, 2)
        ts = phot.photometry(aperture=ap, sky=[in_array[i], out_array[i]])
        ratio, ratio_error, sigma, errb_m = ts.get_ratio(sector=sector)
        label = "sky [" + str(in_array[i]) + "," + str(out_array[i]) + r"]: $area/area_0$" + str(ratio_area_i)
        ts.plot_ratio(label=label, axes=fig, sector=sector, overwrite=True, save="figures/sky.png")
        in_next = in_array[i] + 2
        out_next = constant_area(in_array[0], out_array[0], in_next)
        ratio_area = sp.append(ratio_area, ratio_area_i)
        desv = sp.append(desv, sigma)
        errbar = sp.append(errbar, errb_m)
        if out_next < stamp_rad:
            in_array = sp.append(in_array, in_next)
            out_array = sp.append(out_array, out_next)
            i += 1
        else:
            break
    # plot of fluxes obtained by the iteration
    ax2 = fig.add_subplot(111)
    ax2.locator_params(nbins=7, axis='x')
    ax2.set_title("Ratio at aperture {}, several sky annulus".format(ap))
    lgd = ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig("figures/sky.png", bbox_extra_artists=(lgd,), bbox_inches="tight")
    # plot of standar deviations of data obtained by the iteration
    basic_plot([in_array, in_array], [desv, errbar],
               title=r'Sky annulus vs $\sigma$ and $errorbar$', ylabel="",
               xlabel='Sky external radius', fmt=["o-", "o--"],
               legend=[r"$\sigma$", r"$errorbar$"], save="figures/skystd.png",
               labelsize=labelsize)
    # plot of ratio of areas obtained by the iteration
    basic_plot([ratio_area, ratio_area], [desv, errbar],
               title=r'Relative area vs $\sigma$ and $errorbar$', ylabel="",
               xlabel=r'Relative area $area/area_0$', fmt=['o', '*'],
               legend=[r'$\sigma$', r'$errorbar$'], save="figures/skystdarea.png",
               labelsize=labelsize)
    return in_array[sp.argmin(desv)], out_array[sp.argmin(desv)]


def find_ap(phot, ap_range, sky_test=None, sector=None, labelsize=14):
    """
    finds the optimal aperture for a given range of them.
    :param phot: photometric object from dataproc
    :param ap_range: range of apertures of the form [ap_min, ap_max]
    :param sky_test: test values of photometric sky
    :param sector: sector of the time series out of transit.
    :param labelsize:
    :return: optimal aperture
    """
    matplotlib.rc('xtick', labelsize=labelsize + 4)
    matplotlib.rc('ytick', labelsize=labelsize + 4)

    if sky_test[0] < ap_range[1]+2:
        sky_test[0] = ap_range[1]+2

    # first approximation of optimal aperture
    aps = sp.array([ap_range[0]])
    j = 0
    while aps[j] < ap_range[1]-1:
        aps = sp.append(aps, aps[j]+2)
        j += 1

    desv, opt_ap, errb_0 = get_min_std_ap(phot, aps, sky_test, sector=sector, save="figures/apert.png")
    plot_test(aps, aps, desv, errb_0, title=r"aperture vs $\sigma$ and errorbar", xlabel="aperture", save="figures/apertstd.png")

    # second approximation of optimal aperture
    ap_2 = sp.array([opt_ap - 2, opt_ap - 1, opt_ap, opt_ap + 1, opt_ap + 2])
    desv_2, opt_ap_2, errb_2 = get_min_std_ap(phot, ap_2, sky_test, sector=sector, save="figures/apert2.png")
    plot_test(ap_2, ap_2, desv_2, errb_2, title= r"aperture vs $\sigma$ and errorbar", xlabel="aperture", save="figures/apertstd2.png")

    return opt_ap_2


def find_sky(phot, ap, stamp_rad, sky_range=None, sky_test=None,
             sector=None, labelsize=14):
    """

    :param phot: photometric object from dataproc
    :param ap: aperture value for the timeseries.
    :param stamp_rad: radius of the stamp
    :param sky_range: range of sky values [sky_in_min, sky_out_min, sky_in_max, sky_out_max]
    :param sky_test: test values of photometric sky
    :param sector: sector of the time series out of transit.
    :param labelsize:
    :return: optimal inner and external sky radius.
    """
    matplotlib.rc('xtick', labelsize=labelsize + 3)
    matplotlib.rc('ytick', labelsize=labelsize + 3)

    if sky_range is not None:
        if sky_test[0] > sky_range[1]:
            sky_out_min = sky_test[0] + 2
        else:
            sky_out_min = sky_range[1]
        sky_out_max = sky_range[3]
    else:
        sky_out_min = sky_test[0]+2
        sky_out_max = stamp_rad - 1

    # first approximation of the optimal outer radius
    out_array = sp.array([sky_out_min])
    i = 0
    while out_array[i] <= sky_out_max - 1:
        out_array = sp.append(out_array, out_array[i] + 2)
        i += 1
    desv, opt_sky_out, errb = get_opt_sky_out(phot, ap, sky_test[0],
                                              out_array, sector=sector,
                                              save="figures/sky_outradius.png")
    plot_areas(ap, sky_test[0], out_array, desv, errb,
               save="figures/sky_outradiusstdarea.png", labelsize=labelsize)
    plot_test(out_array, out_array, desv, errb,
              title=r"sky annulus vs $\sigma$ and errorbar",
              xlabel="Sky external radius", save="figures/sky_outradiusstd.png")

    # second approximation of the optimal outer radius
    out_array_2 = sp.array([opt_sky_out - 2, opt_sky_out - 1,
                            opt_sky_out, opt_sky_out + 1, opt_sky_out + 2])
    for sky_out in out_array_2:
        if stamp_rad <= sky_out or sky_out <= sky_test[0]:
            out_array_2 = sp.delete(out_array_2, sp.where(out_array_2 == sky_out))
            if out_array_2[0] - 1 > sky_test[0]:
                out_array_2 = sp.append(out_array_2[0] - 1, out_array_2)
            elif out_array_2[sp.argmax(out_array_2)] + 1 < stamp_rad:
                out_array_2 = sp.append(out_array_2, out_array_2[sp.argmax(out_array_2)] + 1)
    desv_2, opt_sky_out, errb_2 = get_opt_sky_out(phot, ap, sky_test[0], out_array_2, sector=sector,
                                                  save="figures/sky_outradius2.png")
    plot_test(out_array_2, out_array_2, desv_2, errb_2, title=r"sky annulus vs $\sigma$ and errorbar",
              xlabel="Sky external radius", save="figures/sky_outradiusstd2.png")

    if sky_range is not None:
        sky_in_min = sky_range[0]
        if sky_range[2] >= opt_sky_out:
            sky_in_max = opt_sky_out - 1
        else:
            sky_in_max = sky_range[2]
    else:
        sky_in_min = ap
        sky_in_max = opt_sky_out-1

    # first approximation of the optimal inner radius
    in_array = sp.array([sky_in_min])
    i = 0
    while in_array[i] < sky_in_max-1:
        in_array = sp.append(in_array, in_array[i]+2)
        i += 1
    sky_test[1] = opt_sky_out
    desv, opt_sky_in, errb = get_opt_sky_in(phot, ap, in_array,
                                            sky_test=sky_test, sector=sector,
                                            save="figures/sky_inradius.png",
                                            labelsize=labelsize)
    plot_areas(ap, in_array, sky_test[1], desv, errb,
               save="figures/sky_inradiusstdarea.png", labelsize=labelsize)
    plot_test(in_array, in_array, desv, errb,
              title=r"sky annulus vs $\sigma$ and errorbar",
              xlabel="Sky inner radius", save="figures/sky_inradiusstd.png")

    # second approximation of the optimal inner radius
    in_array_2 = sp.array([opt_sky_in-2, opt_sky_in-1, opt_sky_in, opt_sky_in+1, opt_sky_in+2])
    for sky_in in in_array_2:
        if sky_in < ap:
            in_array_2 = sp.delete(in_array_2, 0)
            if in_array_2[len(in_array_2)-1] + 1 < sky_test[1]:
                in_array_2 = sp.append(in_array_2, in_array_2[len(in_array_2)-1] + 1)
        if sky_in >= sky_test[1]:
            in_array_2 = sp.delete(in_array_2, sp.where(in_array_2 == sky_in)[0][0])
    desv_2, opt_sky_in, errb_2 = get_opt_sky_in(phot, ap, in_array_2,
                                                sky_test=sky_test, sector=sector,
                                                save="figures/sky_inradius2.png",
                                                labelsize=labelsize)
    plot_test(in_array_2, in_array_2, desv_2, errb_2, title=r"sky annulus vs $\sigma$ and errorbar",
              xlabel="Sky inner radius", save="figures/sky_inradiusstd2.png")

    return opt_sky_in, opt_sky_out


def astrodir_files(files_path, object_in_hdr, sort_by='JULDAT', fmt=None,
                   first_frame=None, last_frame=None, telescope=None, auto_trim=None,
                   filter_band="I", hdu=None, hdud=None):
    """
    get the astrodir from the files path, could use the target name,
    first and last frame or both, in the last case, if the images
    have OBJECT header empty, then replace it using the target name.
    :param files_path: path of the files.
    :param target: target name
    :param sort_by: parameter used to sort the datas, use JULDAT by default.
    :param fmt: format of the image, if it is ftsc the HDU to use is 1 (second header)
    :param first_frame: first frame of the time series.
    :param last_frame: last frame of the time series.
    :return: astrodir
    """
    if object_in_hdr is None and (last_frame is None or first_frame is None):
        raise TypeError("user must specify a name in header or "
                        "the time of FIRST and LAST frame "
                        "of the observational data in JULIAN DATE")

    if telescope=="ctio_09m" or telescope=="rem_2011" or telescope=="soar" or telescope=='swope': 
        files = dp.AstroDir(files_path, hdu=hdu, hdud=hdud, auto_trim=auto_trim)

        if telescope == "soar" or telescope == "swope":
            if telescope == "soar":
                merge = ["date-obs","time-obs"]
            elif telescope == "swope":
                merge = ["ut-date","ut-time"]
            
            files.jd_from_ut(source=merge)
            files.sort(sort_by)
            files = super_filter(files,hdu=hdu)
    
        else:
            files.jd_from_ut()
            files.sort(sort_by)
            files = super_filter(files)
    else:
        files = dp.AstroDir(files_path,hdu=hdu, auto_trim=auto_trim).sort(sort_by)

        #files = super_filter(files)

    #if fmt == "ftsc":
    #    hdu = 1
    #    files = dp.AstroDir(files_path, hdu=hdu).sort(sort_by)

    #if fmt == "fits" and telescope == "danish":
    #    hdu = 0
    #    files = dp.AstroDir(files_path, hdu=hdu, auto_trim="TRIMSEC").sort(sort_by)

        # check for wrong fits
    #    files = super_filter(files)

    #if fmt == "fits" and telesope != "danish":
    #    hdu = 0
    #    files = dp.AstroDir(files_path, hdu=hdu).sort(sort_by)
    #    files = super_filter(files)

    if last_frame is not None:
        files = files.filter(JULDAT_lt=last_frame + 0.0005)
    if first_frame is not None:
        files = files.filter(JULDAT_gt=first_frame - 0.0005)

    if object_in_hdr is not None:
        if all([files.getheaderval("OBJECT")[x] == '' for x in range(len(files))]):
            files = files.setheader(OBJECT=object_in_hdr)
    files = files.filter(OBJECT=object_in_hdr)

    return files


def super_filter(files,hdu=0):
    '''
    Generate a AstroDir with the good files (data not None).
    :param files:
    :return: files without the wrong fits
    '''
    good = []

    # TODO: return files.filter(naxis_not=0)
    for file in files:
        fit = pf.open(file.filename)
        if fit[hdu].data is not None:
            good.append(file)
        fit.close()
    files_filtered = dp.AstroDir(good[i] for i in range(len(good)))
    return files_filtered



def norm_mflat(calib_path, fmt=None, mbias=0.0, filter_band = "I", flat_in_hdr="FLAT",
               bias_in_hdr="BIAS", imtype_in_hdr="IMAGETYP", filter_in_hdr="FILTB", auto_trim=None, hdu=None, hdud=None):
    """
    generate the AstroDir and the master flat (mflat) normalized of the images
    corresponding to the target name, if the files doesn't have a target name in header,
    it will filter the files by the julian date of the first and the las frame.
    This function only works for images format ".ftsc" and ".fts".
    :param calib_path: path of the calibration files.
    :param fmt: format of the image, if it is ftsc the HDU to use is 1 (second header)
    :param mbias: master bias, 0 by default (no master bias)
    :param filter_band: astronomical filter used, I (infrared) by default
    :return: master flat normalized
    """
    if not fmt == "fts" and not fmt == "ftsc" and not fmt == "fits":
         default_logger.warning("this pipeline only works with files extensions "
                                ".fts, .ftsc or .fits, for other formats it could not works")

    if not imtype_in_hdr == "IMAGETYP" and not imtype_in_hdr == "IMGTYPE":
        default_logger.warning("This pipeline only supports headers with IMAGETYP or IMGTYPE")

    if imtype_in_hdr == "EXPTYPE":
        bias = dp.AstroDir(calib_path,auto_trim=auto_trim).filter(EXPTYPE=bias_in_hdr)
        flats = dp.AstroDir(calib_path, mbias=mbias, hdu=hdu, hdud=hdud, auto_trim=auto_trim,
                            mbias_header=bias[0].readheader()).filter(EXPTYPE=flat_in_hdr)

    if imtype_in_hdr == "OBSTYPE":
        bias = dp.AstroDir(calib_path, auto_trim=auto_trim, hdu=hdu).filter(OBSTYPE=bias_in_hdr)
        flats = dp.AstroDir(calib_path, mbias=mbias, hdu=hdu, hdud=hdud, auto_trim=auto_trim,
                            mbias_header=bias[0].readheader()).filter(OBSTYPE=flat_in_hdr)

    if imtype_in_hdr == "IMAGETYP":
        hdu=0
        bias = dp.AstroDir(calib_path, auto_trim=auto_trim).filter(IMAGETYP=bias_in_hdr)
        flats = dp.AstroDir(calib_path, mbias=mbias, hdu=hdu, hdud=hdud, auto_trim=auto_trim,
                            mbias_header=bias[0].readheader()).filter(IMAGETYP=flat_in_hdr)

    if imtype_in_hdr == "IMGTYPE":
        bias = dp.AstroDir(calib_path, auto_trim=auto_trim).filter(IMGTYPE=bias_in_hdr)
        flats = dp.AstroDir(calib_path, mbias=mbias, hdu=hdu, hdud=hdud, auto_trim=auto_trim,
                            mbias_header=bias[0].readheader()).filter(IMGTYPE=flat_in_hdr)

    mflat_normalized = 1.0

    if filter_in_hdr == "FILTER":
        if len(flats.filter(FILTER=filter_band)) == 1:
            mflat = flats.filter(FILTER=filter_band)[0].reader()
            mflat_normalized = mflat / sp.median(mflat)
        elif len(flats.filter(FILTER=filter_band)) > 1:
            mflat_normalized = flats.filter(FILTER=filter_band).median(normalize=True)
        return mflat_normalized

    if filter_in_hdr == "FILTB":
        if len(flats.filter(FILTB=filter_band)) == 1:
            mflat = flats.filter(FILTB=filter_band)[0].reader()
            mflat_normalized = mflat / sp.median(mflat)
        elif len(flats.filter(FILTB=filter_band)) > 1:
            mflat_normalized = flats.filter(FILTB=filter_band).median(normalize=True)
        #mflat_normalized=super_filter(mflat_normalized)
        return mflat_normalized

    if filter_in_hdr == "FILTERID":
        if len(flats.filter(FILTERID=filter_band)) == 1:
            mflat = flats.filter(FILTERID=filter_band)[0].reader()
            mflat_normalized = mflat / sp.median(mflat)
        elif len(flats.filter(FILTERID=filter_band)) > 1:
            mflat_normalized = flats.filter(FILTERID=filter_band).median(normalize=True)
        return mflat_normalized

    if filter_in_hdr == "FILTER2":
        if len(flats.filter(FILTER2=filter_band)) == 1:
            mflat = flats.filter(FILTER2=filter_band)[0].reader()
            mflat_normalized = mflat / sp.median(mflat)
        elif len(flats.filter(FILTER2=filter_band)) > 1:
            mflat_normalized = flats.filter(FILTER2=filter_band).median(normalize=True)
        return mflat_normalized

    if filter_in_hdr == "WHEEL2":
        if len(flats.filter(WHEEL2=filter_band)) == 1:
            mflat = flats.filter(WHEEL2=filter_band)[0].reader()
            mflat_normalized = mflat / sp.median(mflat)
        elif len(flats.filter(WHEEL2=filter_band)) > 1:
            mflat_normalized = flats.filter(WHEEL2=filter_band).median(normalize=True)
        return mflat_normalized

    #if fmt=="fts":
    #    hdu=0
    #    flat_name="illum"
    #    bias = dp.AstroDir(calib_path).filter(IMAGETYP="zero")
    #    flats = dp.AstroDir(calib_path, mbias=mbias, hdu=hdu,mbias_header=bias[0].readheader()).filter(IMAGETYP=flat_name)
        #pdb.set_trace()

    #    mflat_normalized = 1.0
    #    if len(flats.filter(FILTER=filter_band)) == 1:
    #        mflat = flats.filter(FILTER=filter_band)[0].reader()
    #        mflat_normalized = mflat / sp.median(mflat)
    #        #pdb.set_trace()
    #    elif len(flats.filter(FILTER=filter_band)) > 1:
    #        mflat_normalized = flats.filter(FILTER=filter_band).median(normalize=True)
           #pdb.set_trace()
    #   return mflat_normalized

    #if fmt == "ftsc":
    #    hdu = 1
    #    flat_name="flat"

    #   flats = dp.AstroDir(calib_path, mbias=mbias, hdu=hdu).filter(IMAGETYP=flat_name)
    #    mflat_normalized = 1.0
    #    if len(flats.filter(FILTER=filter_band)) == 1:
    #        mflat = flats.filter(FILTER=filter_band)[0].reader()
    #        mflat_normalized = mflat / sp.median(mflat)
    #    elif len(flats.filter(FILTER=filter_band)) > 1:
    #        mflat_normalized = flats.filter(FILTER=filter_band).median(normalize=True)

    # #    return mflat_normalized
    #
    # if fmt == "fits":
    #     hdu=0
    #     #flats_dome = dp.AstroDir(calib_path, mbias=mbias, hdu=hdu).filter(IMAGETYP="FLAT").filter(OBJECT="domeflat").sort("JD")
    #     bias = dp.AstroDir(calib_path, auto_trim="TRIMSEC").filter(IMAGETYP="BIAS")
    #     bias = super_filter(bias)
    #     flats = dp.AstroDir(calib_path, mbias=mbias, hdu=hdu, auto_trim="TRIMSEC", mbias_header=bias[0].readheader()).filter(IMAGETYP="FLAT")
    #     flats = super_filter(flats)
    #     mflat_normalized = 1.0
    #     if len(flats.filter(FILTB=filter_band)) == 1:
    #
    #         mflat = flats.filter(FILTB=filter_band)[0].reader()
    #         mflat_normalized = mflat/sp.median(mflat)
    #     elif len(flats.filter(FILTB=filter_band)) > 1:
    #         mflat_normalized = flats.filter(FILTB=filter_band).median(normalize=True)
    #
    #     return mflat_normalized

def master_bias(calib_path, fmt=None, bias_in_hdr="BIAS", imtype_in_hdr="IMAGETYP", hdu=None, hdud=None, exposure=None):
    """
    generate the master bias
    :param calib_path: path of the calibration files.
    :param fmt: format of the image
    :return: master bias
    """
    mbias = 0.0
    #pdb.set_trace()
    if imtype_in_hdr == "OBSTYPE":
        bias = dp.AstroDir(calib_path, hdu=hdu,hdud=hdud).filter(OBSTYPE=bias_in_hdr)

    if imtype_in_hdr=="IMAGETYP":
        bias = dp.AstroDir(calib_path).filter(IMAGETYP=bias_in_hdr)

    if imtype_in_hdr=="IMGTYPE":
        bias = dp.AstroDir(calib_path).filter(IMGTYPE=bias_in_hdr)

    if imtype_in_hdr=="EXPTYPE":
        bias = dp.AstroDir(calib_path).filter(EXPTYPE=bias_in_hdr)

    #if fmt=="fts" or fmt=="ftsc":
    #    bias = dp.AstroDir(calib_path).filter(IMAGETYP="zero")

    #if fmt == "fits":
    #    bias = dp.AstroDir(calib_path).filter(IMAGETYP="BIAS")

    #bias = super_filter(bias)
    if len(bias) == 1:
        mbias = bias[0].reader()
    
    if len(bias) > 1:
        if exposure is not None:
            mbias=bias.lin_interp(target=exposure)
        else:
            mbias = bias.median()
    return mbias


def plot_test(x1, x2, y1, y2, label1=r"$\sigma$", label2="errorbar",
              xlabel="x", title="plot", fmt1='o-', fmt2='o--', save=None):
    """
    auxiliar funtion to plot two arrays of datas (y-axis) with the same x-axis,
    for this pipeline it is used to plot standard deviation and mean error bar vs
    apertures, skies inner or external radius and area.
    :param x1:
    :param x2:
    :param y1:
    :param y2:
    :param label1: legend of the x1, y1 datas, by default it is sigma (std. deviation)
    :param label2: legend of the x2, y2 datas, by default it is errorbar
    :param xlabel:
    :param title:
    :param fmt1: format of the plot for x1,y1 datas, by default it is 'o-'
    :param fmt2: format of the plot for x2,y2 datas, by default it is 'o--'
                 to diferenciate from x1,y1 datas
    :param save:
    :return:
    """
    fig, ax = plt.subplots()
    ax.plot(x1, y1, fmt1, label=label1)
    ax.plot(x2, y2, fmt2, label=label2)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(xlabel)

    if save is not None:
        plt.tight_layout()
        plt.savefig(save)
    else:
        plt.show()


def find_format(files_path):
    """
    find the format of the images using the basename.
    :param files_path: path of the files
    :return: the format (like "ftsc" or "fts")
    """
    f = dp.AstroDir(files_path)
    basename = f[0].getheaderval("basename")
    ext = ["zip","ZIP","gz","GZ"]
    if basename[0].split(".")[-1] in ext:
        return basename[0].split(".")[-2]
    else:
        return basename[0].split(".")[-1]


def constant_area(in_1, out_1, in_2):
    """
    giving a first value of inner and external sky radius, and a second value of a inner
    sky radius, calculates the necessary value of second external sky radius to keep
    the area approximately constant.
    :param in_1: first inner sky radius
    :param out_1: first external sky radius
    :param in_2: second inner sky radius
    :return: second external sky radius.
    """
    return int(sp.sqrt(out_1**2 - in_1**2 + in_2**2))


def basic_plot(x, y, axes=None, overwrite=False, title="title",
               xlabel="x", ylabel="y", fmt=None, legend=None,
               save=None, labelsize=14):
    """
    auxiliar function to plot datas.
    :param x:
    :param y:
    :param axes:
    :param overwrite:
    :param title:
    :param xlabel:
    :param ylabel:
    :param fmt:
    :param legend:
    :param save:
    :return:
    """
    matplotlib.rc('axes', titlesize=labelsize + 3)
    matplotlib.rc('axes', labelsize=labelsize + 3)

    fig, ax = dp.figaxes(axes, overwrite=overwrite)
    ax.cla()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for x_i, y_i, fmt_i, legend_i in zip(x, y, fmt, legend):
        ax.plot(x_i, y_i, fmt_i, label=legend_i)
    ax.legend(loc=1)
    plt.tight_layout()
    if save is not None:
        plt.savefig(save)
    else:
        plt.show()


def areas(ap, sky_in, sky_out):
    """
    calculate the area ratio of sky_area/ap_area
    :param ap: aperture
    :param sky_in: inner sky radius
    :param sky_out: external sky radius
    :return: area ratio
    """
    return ((sky_out**2 - sky_in**2) * 1.0) / ((ap**2) * 1.0)


def plot_areas(ap, sky_in_range, sky_out, desv, errorbar, save=None,
               labelsize=14):
    """
    plot standard deviation and mean errorbar vs area ratio
    :param ap: aperture
    :param sky_in_range: range of inner skies radius.
    :param sky_out: external sky radius
    :param desv: standard deviation
    :param errorbar: mean errorbar
    :param save: name to save the plot
    :return:
    """
    matplotlib.rc('xtick', labelsize=labelsize + 4)
    matplotlib.rc('ytick', labelsize=labelsize + 4)
    razon = sp.array([])
    if isinstance(sky_in_range, sp.ndarray):
        area_0 = areas(ap, sky_in_range[0], sky_out)
        for sky_in in sky_in_range:
            area = areas(ap, sky_in, sky_out)
            r = area / area_0
            razon = sp.append(razon, r)
    elif isinstance(sky_out, sp.ndarray):
        area_0 = areas(ap, sky_in_range, sky_out[0])
        for sky_o in sky_out:
            area = areas(ap, sky_in_range, sky_o)
            r = area / area_0
            razon = sp.append(razon, r)

    fig, ax = plt.subplots()
    ax.plot(razon, desv, 'o', label=r"$\sigma$")
    ax.plot(razon, errorbar, 'o', label=r"$errorbar$")
    ax.legend(fontsize='x-small')
    ax.set_title(r"Relative area vs $\sigma$ and $errorbar$")
    ax.set_xlabel(r"relative area $area/area_0$")
    if save is not None:
        plt.tight_layout()
        plt.savefig(save)
    else:
        plt.show()

    return


def exptime(files_dir, save=None, labelsize=14, telescope=None):
    """
    plot the exposure time per frame
    :param files_dir:
    :param save:
    :return:
    """
    matplotlib.rc('axes', titlesize=labelsize + 6)
    matplotlib.rc('axes', labelsize=labelsize + 5)
    times = files_dir.getheaderval("exptime")
    if telescope == "warsaw":
        jd = files_dir.getheaderval("juldat")
    else:
        jd = files_dir.getheaderval("JD")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(jd, times, 'o--', label="exposition time")
    ax.set_xlabel("JD")
    ax.locator_params(nbins=7, axis='x')
    ax.set_ylabel("time of exposure [seg]")
    ax.set_title("Exptime used in every frame")
    plt.tight_layout()
    if save is not None:
        plt.savefig(save)
    else:
        plt.show()


def check_dirs(dir, Path='./'):
    """
    check if the directory exists and if it's not, then create it
    :param dir:
    :param path:
    :return:
    """
    if not os.path.exists(Path + dir):
        os.makedirs(Path + dir)


def symlink_file(file, ref_path=None, ref_symlink=None, Path='./'):
    """
    check if the symlink exists and if it's not, then create it using a reference
    symlink from the same path or a reference path of origin.
    :param file:
    :param ref_path:
    :param ref_symlink:
    :param Path:
    :return:
    """
    if not os.path.exists(Path + file):
        if ref_path is not None and os.path.exists(ref_path + "extraction.tex"):
            os.system("ln -s " + ref_path + "extraction.tex " + Path + file)
        if ref_symlink is not None and os.path.exists(ref_symlink):
            realpath = os.path.realpath(ref_symlink)
            head, tail = os.path.split(realpath)
            if os.path.exists(head + "extraction.tex"):
                os.system("ln -s " + head + "extraction.tex " + Path + file)
            else:
                raise IOError("cannot find the extraction template of the reference path {},"
                              "make sure that you have the file in the same directory that pipeline"
                              "or create the symlink by yourself.".format(head))
        else:
            raise IOError("cannot create the symlink to the extraction template, file extraction.tex"
                          "not found")

def merge(files_path,calib_path, hdud):
    """
    Merges the hdu sub-images of the files contained on each
    specified folder
    """
    print("Merging hdu subimages. ",end='')
    link = [calib_path, files_path]
    for path in link:
        files = glob.glob(path)
        for filename in files:
            dp.merger(filename=filename, nhdu = hdud)
    print('done.')

def generate_extraction(target):
    """
    generate te extraction PDF using the symplink.
    :param target:
    :return:
    """
    symlink_file(target + "_extraction.tex", ref_symlink="./log/pipeline.py")

    os.system("pdflatex " + target + "_extraction.tex")
    os.system("pdflatex " + target + "_extraction.tex")

    return


def make_tramos_custom(ap, target, Path='./'):
    """
    create the tramos_custom.sty and if it's exists, then only change
    the planet name (target) and aperture(ap) values.
    :param ap:
    :param target:
    :param Path:
    :return:
    """
    if os.path.exists(Path + 'tramos_custom.sty'):
        with open(Path + 'tramos_custom.sty', 'r') as file:
            data = file.readlines()

        data[6] = '\\newcommand{\\planet}{' + target + '}\n'
        data[9] = '\\newcommand{\\aperture}{' + str(ap) + '}\n'

        with open(Path + 'tramos_custom.sty', 'w') as file:
            file.writelines(data)

    else:
        data = ['\\NeedsTeXFormat{LaTeX2e}\n', '\\usepackage{listings}\n',
                '\n', '\\ProvidesPackage{tramos_custom}[2017/04/29 first release]\n',
                '\n', '\n', '\\newcommand{\\planet}{' + target + '}\n',
                '\\newcommand{\\inst}{Warsaw}\n', '\\newcommand{\\epoch}{@EPOCH@}\n',
                '\\newcommand{\\aperture}{' + ap + '}\n', '\\newcommand{\\period}{@PERIOD@}\n',
                '\\newcommand{\\sma}{@SMA@}\n', '\\newcommand{\\ra}{@RA@}\n',
                '\\newcommand{\\dec}{@DEC@}\n', '\\newcommand{\\magv}{@MAGV@}\n',
                '\\newcommand{\\pipeline}{\\lstinputlisting[language=Python]{log/runme.py}}\n',
                '\\newcommand{\\comments}{}\n', '\n', '\\endinput']

        with open(Path + 'tramos_custom.sty', 'w') as file:
            file.writelines(data)


# TODO: NOT IMPLEMENTED
def get_min_std_ap_v2(phot, aps, skys, sky_test=None, i=0, sector=None, save=None):
    """

    :param phot:
    :param aps:
    :param skys:
    :param sky_test:
    :param i:
    :param sector:
    :param save:
    :return:
    """
    desv = sp.array([])
    errb = sp.array([])
    if isinstance(aps, sp.ndarray):
        fig = plt.figure()
        for aperture in aps:
            ts = phot.photometry(aperture=aperture, sky=skys)
            ratio, ratio_error, sigma, errb_m = ts.get_ratio(sector=sector)
            label=','.join(["apert" + str(aperture),
                            r" $\sigma$" + str(round(sigma, 3)),
                            r" $errorbar$" + str(round(errb_m, 3))])
            ts.plot_ratio(label=label, axes=fig, sector=sector, overwrite=True, save=save)
            desv = sp.append(desv, sigma)
            errb = sp.append(errb, errb_m)
            if len(desv) >=3 and (desv.argmin() == len(desv) - 3):
                break

        ax = fig.add_subplot(111)
        lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if save is not None:
            plt.savefig(save, bbox_extra_artists=(lgd,), bbox_inches="tight")
        else:
            plt.show()
        opt_ap = aps[sp.argmin(desv)]

        return desv, opt_ap, errb


def find_ap_v2(phot, ap_range, stamp_rad, sky_test=None, sector=None, save=None):
    """

    :param phot:
    :param ap_range:
    :param stamp_rad:
    :param sky_test:
    :param sector:
    :param save:
    :return:
    """
    matplotlib.rc('xtick', labelsize=20)
    matplotlib.rc('ytick', labelsize=20)
    save1 = save + "/apert.png"
    save2 = save + "/apert2.png"
    save3 = save + "/apertstd.png"
    save4 = save + "/apertstd2.png"
    if sky_test[0] < ap_range[1]:
        sky_test[0] = ap_range[1]

    # unique approximation of optimal aperture
    aps = sp.array([ap_range[0]])
    j=0
    while (aps[j] < ap_range[1]-1):
        aps = sp.append(aps, aps[j]+1)
        j = j+1

    desv, opt_ap, errb_0 = get_min_std_ap_v2(phot, aps, sky_test, sector=sector, save=save1)
    plot_test(aps, aps, desv, errb_0, title=r"aperture vs $\sigma$ and errorbar", xlabel="aperture", save=save3)

    return opt_ap

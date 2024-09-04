"""Linear drift rate measurements using a 2D auto-correlation analysis
and Monte Carlo resampling.

"""

import copy
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.signal

from .dfdt_utils import DynamicSpectrum, find_burst, dedisperse, add_text


# suppress warnings related to having NaNs in dynamic spectra
warnings.filterwarnings(action="ignore",
                        message="Degrees of freedom <= 0 for slice.")
warnings.filterwarnings(action="ignore", message="Mean of empty slice")
warnings.filterwarnings(action="ignore",
                        message="invalid value encountered in less")
warnings.filterwarnings(action="ignore",
                        message="invalid value encountered in greater")


def gauss_2d(xy, *p):
    """2D gaussian."""
    x, y = xy
    amplitude, sigma_x, sigma_y, theta, offset = p

    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) \
        / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) \
        / (4 * sigma_y ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) \
        / (2 * sigma_y ** 2)
    z = offset + amplitude * np.exp(-(a * (x ** 2) + 2 * b * x * y + c \
        * (y ** 2)))

    return z.ravel()


def ac_mc_drift(
    dedispersed_intensity,
    dm_uncertainty,
    source,
    eventid,
    ds,
    sub_factor=64,
    dm_trials=10,
    mc_trials=10,
    detection_confidence=90.0,
    uncertainty_confidence=68.0,
    plot_result=True,
    plot_all=False,
    peak=None,
    width=None,
    fdir="./results/",
    rfi_cleaning=True,
    normalize=True
):
    """Measure linear drift rate with a 2D auto-correlation method and
    uncertainties with Monte Carlo resampling.

    Parameters
    ----------
    dedispersed_intensity : array_like
        Dedispersed waterfall.
    dm_uncertainty : float
        Statistical uncertainty on DM, in pc cm-3, for resampling.
    source : str
        Source name, used for plotting purposes.
    eventid : str
        CHIME/FRB event ID, used for plotting purposes.
    ds : :obj:DynamicSpectrum
        Object holding intensity data parameters (i.e., time/frequency
        resolution).
    sub_factor : int
        Factor to subband intensity data by, 64 by default.
    dm_trials : int
        Number of DM trials, 10 by default. If 1, do not resample the
        DM at all.
    mc_trials : int
        Number of Monte Carlo trials, 10 by default. If 1, do not
        resample the noise at all.
    detection_confidence : float
        Confidence interval in percent to calculate for results and to
        display on plot (i.e. how many samples need to have positive/
        negative drift rate to claim a confident detection), 90 by default.
    uncertainty_confidence : float
        Confidence interval in percent to calculate uncertainty region
        for results, 68 (1sigma) by default.
    plot_result : bool
        Plot analysis results, True by default.
    plot_all : bool
        Plot all resampled 2D autocorrelations (for debuggin), False by
        default.
    peak : int, optional
        Pulse peak position index. None by default.
    width : int, optional
        Pulse width, as a factor of sampling time. None by default.

    Returns
    -------
    constrained : bool
        Is measurement constrained or not? I.e., do all thetas fall in
        the same quadrant.
    dfdt_data : float
        Linear drift rate from data, in MHz/ms.
    dfdt_mc : float
        Mean linear drift rate from MC trials, in MHz/ms.
    dfdt_mc_low : float
        Linear drift rate lower bound on containment interval from MC
        trials, in MHz/ms.
    dfdt_mc_high : float
        Linear drift rate upper bound on containment interval from MC
        trials, in MHz/ms.

    """
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    print("{} -- {} -- Analyzing..".format(source, eventid))

    # mask out top channel (if not already masked)
    dedispersed_intensity[0, ...] = np.nan

    if rfi_cleaning:

        # mask out all outliers 3 sigma away from the channel mean
        channel_means = np.nanmean(dedispersed_intensity, axis=1)
        channel_stds = np.nanstd(dedispersed_intensity, axis=1)
        threshold = np.repeat(
            channel_means - 3 * channel_stds, dedispersed_intensity.shape[1]
        ).reshape(dedispersed_intensity.shape)
        dedispersed_intensity[dedispersed_intensity < threshold] = np.nan
        threshold = np.repeat(
            channel_means + 3 * channel_stds, dedispersed_intensity.shape[1]
        ).reshape(dedispersed_intensity.shape)
        dedispersed_intensity[dedispersed_intensity > threshold] = np.nan

    if normalize:
        # subtract mean (can also try median)
        mean = np.nanmean(dedispersed_intensity, axis=1)
        mean = np.repeat(mean, dedispersed_intensity.shape[1]).reshape(
            dedispersed_intensity.shape
        )

        dedispersed_intensity = dedispersed_intensity - mean

    subbanded_channel_bw = (
        ds.df_mhz * (ds.nchan / dedispersed_intensity.shape[0])
    )
    center_frequencies = (
        np.arange(
            ds.freq_bottom_mhz,
            ds.freq_top_mhz,
            subbanded_channel_bw,
        )
        + subbanded_channel_bw / 2.0
    )

    # calculate drift rate from data at best known DM
    intensity = copy.deepcopy(dedispersed_intensity)

    if peak is None or width is None:
        ts = np.nansum(intensity, axis=0)
        peak, width, snr = find_burst(ts, width_factor=4, plot = False)

    window = 100

    # increase window for wide bursts
    while width > 0.5 * window:
        window += 100

    sub = np.nanmean(
        intensity.reshape(-1, sub_factor, intensity.shape[1]), axis=1)
    median = np.nanmedian(sub)
    sub[sub == 0.0] = median
    sub[np.isnan(sub)] = median

    waterfall = copy.deepcopy(sub[..., peak - window // 2 : peak + window // 2])

    # select noise before (and after) the burst (if necessary)
    #noise_window = (peak - 3 * window // 2, peak - window // 2)
    noise_window = (peak - 4 * window // 2, peak - 2 * window // 2)

    if noise_window[0] < 0:
        difference = abs(noise_window[0])
        noise_window = (noise_window[0] + difference,
                        noise_window[1] + difference)
        noise_waterfall = copy.deepcopy(np.roll(
            sub, difference, axis=1)[...,noise_window[0]:noise_window[1]]
        )
    else:
        noise_waterfall = copy.deepcopy(
            sub[...,noise_window[0] : noise_window[1]]
        )

    ac2d = scipy.signal.correlate2d(
        waterfall, waterfall, mode="full", boundary="fill", fillvalue=0
    )

    ac2d[ac2d.shape[0] // 2, :] = np.nan
    ac2d[:, ac2d.shape[1] // 2] = np.nan

    scaled_ac2d = copy.deepcopy(ac2d)

    scaling_factor = np.nanmax(scaled_ac2d)
    scaled_ac2d = scaled_ac2d / scaling_factor

    noise_ac2d = scipy.signal.correlate2d(
        noise_waterfall, noise_waterfall, mode="full", boundary="fill",
        fillvalue=0
    )

    noise_ac2d[noise_ac2d.shape[0] // 2, :] = np.nan
    noise_ac2d[:, noise_ac2d.shape[1] // 2] = np.nan

    scaled_noise_ac2d = copy.deepcopy(noise_ac2d)

    scaled_noise_ac2d = scaled_noise_ac2d / scaling_factor

    dts = (
        np.arange(-ac2d.shape[1] / 2 + 1, ac2d.shape[1] / 2 + 1)
        * ds.dt_s * 1e3
    )

    dfs = (
        np.arange(-ac2d.shape[0] / 2 + 1, ac2d.shape[0] / 2 + 1)
        * ds.df_mhz
        * (ds.nchan / dedispersed_intensity.shape[0])
        * sub_factor
    )

    # construct data model
    nanmask = np.isnan(scaled_ac2d)
    x, y = [arr.T for arr in np.meshgrid(dfs, dts)]

    time_width = width * ds.dt_s * 1e3
    p0 = np.nanmax(scaled_ac2d), 45, time_width, 0.2, np.nanmedian(scaled_noise_ac2d)
    try:
        p1, pcov = scipy.optimize.curve_fit(
            gauss_2d, (x[~nanmask], y[~nanmask]),
            scaled_ac2d[~nanmask].flatten(), p0=p0
        )

        # let theta range from -pi to pi
        theta = p1[-2] % np.pi
        theta_sigma = np.sqrt(np.diag(pcov))[-2]

        # rotate theta for drift rate calculation by 90 deg
        sigma_x = p1[1]
        sigma_y = p1[2]
        if sigma_y > sigma_x:
            theta += np.pi / 2.
        dfdt_data = 1.0 / np.tan(-theta)
    except:
        # fall back on fit guess if fit did not converge
        p1 = p0

        theta = np.nan
        theta_sigma = np.nan
        dfdt_data = np.nan

    print("{} -- {} -- df/dt (data) = {:.2f} MHz/ms".format(
        source, eventid, dfdt_data))

    # construct noise model
    dim = waterfall.shape

    mus = np.nanmean(scaled_noise_ac2d[...,dim[1] // 2 : -dim[1] // 2 + 1],
        axis=1)
    sigmas = np.nanstd(scaled_noise_ac2d[..., dim[1] // 2 : -dim[1] // 2 + 1],
        axis=1)

    # replace NaNs in center with average of neighboring channels
    mus[mus.shape[0] // 2] = np.nanmean(
        mus[mus.shape[0] // 2 - 1 : mus.shape[0] // 2 + 2]
    )
    sigmas[sigmas.shape[0] // 2] = np.nanmean(
        sigmas[sigmas.shape[0] // 2 - 1 : sigmas.shape[0] // 2 + 2]
    )

    noise_model = np.random.normal(
        loc=mus,
        scale=sigmas,
        size=(scaled_noise_ac2d.shape[1], np.broadcast(mus, sigmas).size),
    ).T

    data = copy.deepcopy(scaled_ac2d)
    noise = copy.deepcopy(scaled_noise_ac2d)

    model = gauss_2d((x, y), *p1).reshape(x.shape)

    fig, ax = plt.subplots(4, 2, figsize=(8.5, 11), sharey="row")

    # data analysis
    ax[0, 0].set_title("Burst waterfall")
    im = ax[0, 0].imshow(
        waterfall,
        aspect="auto",
        interpolation="none",
        origin="lower",
        cmap="viridis",
        extent=(
            -window / 2 * ds.dt_s * 1e3,
            window / 2 * ds.dt_s * 1e3,
            ds.freq_bottom_mhz,
            ds.freq_top_mhz,
        ),
    )
    fig.colorbar(im, ax=ax[0, 0])
    add_text(ax[0, 0], eventid, color="white")
    ax[0, 0].set_xlabel("Time (ms)")
    ax[0, 0].set_ylabel("Observing frequency (MHz)")

    ax[1, 0].set_title("Burst 2D auto-correlation data")
    im = ax[1, 0].imshow(
        data,
        aspect="auto",
        interpolation="none",
        origin="lower",
        cmap="viridis",
        extent=(
            min(dts) - np.diff(dts)[0],
            max(dts) + np.diff(dts)[0],
            min(dfs) - np.diff(dfs)[0],
            max(dfs) + np.diff(dfs)[0],
        ),
    )
    fig.colorbar(im, ax=ax[1, 0])
    ax[1, 0].set_xlabel("$\Delta$t (ms)")
    ax[1, 0].set_ylabel("$\Delta$f (MHz)")

    ax[2, 0].set_title("Burst 2D auto-correlation model")
    im = ax[2, 0].imshow(
        model,
        aspect="auto",
        interpolation="none",
        origin="lower",
        cmap="viridis",
        extent=(
            min(dts) - np.diff(dts)[0],
            max(dts) + np.diff(dts)[0],
            min(dfs) - np.diff(dfs)[0],
            max(dfs) + np.diff(dfs)[0],
        ),
    )
    fig.colorbar(im, ax=ax[2, 0])
    add_text(
        ax[2, 0],
        r"$\theta$ = " + "{:.3f} rad\ndf/dt = {:.1f} MHz/ms".format(
            theta, dfdt_data),
        color="white",
    )
    ax[2, 0].set_xlabel("$\Delta$t (ms)")
    ax[2, 0].set_ylabel("$\Delta$f (MHz)")

    res_med = np.nanmedian(data - model)
    res_std = np.nanstd(data - model)
    vmin = res_med - 3 * res_std
    vmax = res_med + 3 * res_std

    ax[3, 0].set_title("Burst 2D auto-correlation residuals")
    im = ax[3, 0].imshow(
        data - model,
        aspect="auto",
        interpolation="none",
        origin="lower",
        cmap="bwr",
        vmin=vmin,
        vmax=vmax,
        extent=(
            min(dts) - np.diff(dts)[0],
            max(dts) + np.diff(dts)[0],
            min(dfs) - np.diff(dfs)[0],
            max(dfs) + np.diff(dfs)[0],
        ),
    )
    fig.colorbar(im, ax=ax[3, 0])
    add_text(ax[3, 0], "Capped at $\pm3\sigma$ from median")
    ax[3, 0].set_xlabel("$\Delta$t (ms)")
    ax[3, 0].set_ylabel("$\Delta$f (MHz)")

    # noise analysis
    ax[0, 1].set_title("Noise waterfall")
    im = ax[0, 1].imshow(
        (noise_waterfall),
        aspect="auto",
        interpolation="none",
        origin="lower",
        cmap="viridis",
        extent=(
            -window / 2 * ds.dt_s * 1e3,
            window / 2 * ds.dt_s * 1e3,
            ds.freq_bottom_mhz,
            ds.freq_top_mhz,
        ),
    )
    fig.colorbar(im, ax=ax[0, 1])
    ax[0, 1].set_xlabel("Time (ms)")

    ax[1, 1].set_title("Noise 2D auto-correlation data")
    im = ax[1, 1].imshow(
        noise,
        aspect="auto",
        interpolation="none",
        origin="lower",
        cmap="viridis",
        extent=(
            min(dts) - np.diff(dts)[0],
            max(dts) + np.diff(dts)[0],
            min(dfs) - np.diff(dfs)[0],
            max(dfs) + np.diff(dfs)[0],
        ),
    )
    ax[1, 1].axvspan(
        min(dts), min(dts) / 2, ls=":", hatch="//", edgecolor="white",
        facecolor="none"
    )
    ax[1, 1].axvspan(
        max(dts) / 2, max(dts), ls=":", hatch="//", edgecolor="white",
        facecolor="none"
    )
    fig.colorbar(im, ax=ax[1, 1])
    ax[1, 1].set_xlabel("$\Delta$t (ms)")

    ax[2, 1].set_title("Noise 2D auto-correlation model")
    im = ax[2, 1].imshow(
        noise_model,
        aspect="auto",
        interpolation="none",
        origin="lower",
        cmap="viridis",
        extent=(
            min(dts) - np.diff(dts)[0],
            max(dts) + np.diff(dts)[0],
            min(dfs) - np.diff(dfs)[0],
            max(dfs) + np.diff(dfs)[0],
        ),
    )
    fig.colorbar(im, ax=ax[2, 1])
    ax[2, 1].set_xlabel("$\Delta$t (ms)")

    res_med = np.nanmedian(noise - noise_model)
    res_std = np.nanstd(noise - noise_model)
    vmin = res_med - 3 * res_std
    vmax = res_med + 3 * res_std

    ax[3, 1].set_title("Noise 2D auto-correlation residuals")
    im = ax[3, 1].imshow(
        noise - noise_model,
        aspect="auto",
        interpolation="none",
        origin="lower",
        cmap="bwr",
        vmin=vmin,
        vmax=vmax,
        extent=(
            min(dts) - np.diff(dts)[0],
            max(dts) + np.diff(dts)[0],
            min(dfs) - np.diff(dfs)[0],
            max(dfs) + np.diff(dfs)[0],
        ),
    )
    fig.colorbar(im, ax=ax[3, 1])
    add_text(ax[3, 1], "Capped at $\pm3\sigma$ from median")
    ax[3, 1].set_xlabel("$\Delta$t (ms)")

    plt.tight_layout()
    plt.savefig(fdir + "ac_drift_{}.png".format(eventid), dpi=80,
        bbox_inches="tight")
    plt.close()

    # resample data DM and noise distribution

    # draw random DM deviation from uncertainty distribution
    dms = dm_uncertainty * np.random.randn(dm_trials)
    thetas = np.empty(dm_trials * mc_trials)
    theta_sigmas = np.empty(dm_trials * mc_trials)
    drift_rates = np.empty(dm_trials * mc_trials)

    print("{} -- {} -- Resampling data..".format(source, eventid))
    for dm_trial in range(dm_trials):

        intensity = copy.deepcopy(dedispersed_intensity)

        if dm_trials == 1:
            # do not resample at all
            random_dm = 0
        else:
            # resample
            random_dm = dms[dm_trial]

            dedisperse(intensity, center_frequencies,  ds.dt_s, dm=random_dm,
                reference_frequency=((ds.freq_top_mhz - ds.freq_bottom_mhz)/2))

        sub = np.nanmean(intensity.reshape(-1, sub_factor, intensity.shape[1]),
            axis=1)
        median = np.nanmedian(sub)
        sub[sub == 0.0] = median
        sub[np.isnan(sub)] = median

        waterfall = copy.deepcopy(
            sub[..., peak - window // 2 : peak + window // 2])

        # select noise before (and after) the burst (if necessary)
        noise_window = (peak - 3 * window // 2, peak - window // 2)

        if noise_window[0] < 0:
            difference = abs(noise_window[0])
            noise_window = (noise_window[0] + difference,
                            noise_window[1] + difference)
            noise_waterfall = copy.deepcopy(np.roll(
                sub, difference, axis=1)[...,noise_window[0]:noise_window[1]]
            )
        else:
            noise_waterfall = copy.deepcopy(
                sub[...,noise_window[0]:noise_window[1]]
            )

        ac2d = scipy.signal.correlate2d(
            waterfall, waterfall, mode="full", boundary="fill", fillvalue=0
        )

        ac2d[ac2d.shape[0] // 2, :] = np.nan
        ac2d[:, ac2d.shape[1] // 2] = np.nan

        scaled_ac2d = copy.deepcopy(ac2d)

        scaling_factor = np.nanmax(scaled_ac2d)
        scaled_ac2d = scaled_ac2d / scaling_factor

        noise_ac2d = scipy.signal.correlate2d(
            noise_waterfall, noise_waterfall, mode="full", boundary="fill",
            fillvalue=0
        )

        noise_ac2d[noise_ac2d.shape[0] // 2, :] = np.nan
        noise_ac2d[:, noise_ac2d.shape[1] // 2] = np.nan

        scaled_noise_ac2d = copy.deepcopy(noise_ac2d)

        scaled_noise_ac2d = scaled_noise_ac2d / scaling_factor

        dts = (
            np.arange(-ac2d.shape[1] / 2 + 1, ac2d.shape[1] / 2 + 1)
            * ds.dt_s * 1e3
        )
        dfs = (
            np.arange(-ac2d.shape[0] / 2 + 1, ac2d.shape[0] / 2 + 1)
            * ds.df_mhz
            * (ds.nchan / dedispersed_intensity.shape[0])
            * sub_factor
        )

        # construct data model
        nanmask = np.isnan(scaled_ac2d)
        x, y = [arr.T for arr in np.meshgrid(dfs, dts)]

        p0 = np.nanmax(scaled_ac2d), 45, time_width, 0.2, np.nanmedian(scaled_noise_ac2d)
        try:
            p1, pcov = scipy.optimize.curve_fit(
                gauss_2d,
                (x[~nanmask], y[~nanmask]),
                scaled_ac2d[~nanmask].flatten(),
                p0=p0,
            )

            sigma_x = p1[1]
            sigma_y = p1[2]
            if sigma_y > sigma_x:
                theta += np.pi / 2.
        except:
            # fall back on fit guess if fit did not converge
            p1 = p0

        # construct noise model
        dim = waterfall.shape

        mus = np.nanmean(
            scaled_noise_ac2d[..., dim[1] // 2 : -dim[1] // 2 + 1], axis=1
        )
        sigmas = np.nanstd(
            scaled_noise_ac2d[..., dim[1] // 2 : -dim[1] // 2 + 1], axis=1
        )

        # replace NaNs in center with average of neighboring channels
        mus[mus.shape[0] // 2] = np.nanmean(
            mus[mus.shape[0] // 2 - 1 : mus.shape[0] // 2 + 2]
        )
        sigmas[sigmas.shape[0] // 2] = np.nanmean(
            sigmas[sigmas.shape[0] // 2 - 1 : sigmas.shape[0] // 2 + 2]
        )

        data = copy.deepcopy(scaled_ac2d)
        noise = copy.deepcopy(scaled_noise_ac2d)

        model = gauss_2d((x, y), *p1).reshape(x.shape)

        # monte carlo resampling
        for mc_trial in range(mc_trials):

            if mc_trials == 1:
                # do not resample the noise
                random_data = model

            else:
                # resample the noise
                noise_model = np.random.normal(
                    loc=mus,
                    scale=sigmas,
                    size=(noise.shape[1], np.broadcast(mus, sigmas).size),
                ).T

                try:
                    random_data = model + noise_model
                except:
                    thetas[dm_trial * mc_trials + mc_trial] = np.nan
                    theta_sigmas[dm_trial * mc_trials + mc_trial] = np.nan
                    drift_rates[dm_trial * mc_trials + mc_trial] = np.nan
                    continue

            p0 = np.nanmax(scaled_ac2d), 45, time_width, 0.2, np.nanmedian(scaled_noise_ac2d)
            try:
                p1, pcov = scipy.optimize.curve_fit(
                    gauss_2d,
                    (x[~nanmask], y[~nanmask]),
                    random_data[~nanmask].flatten(),
                    p0=p0,
                )
                # let theta range from -pi to pi
                random_theta = p1[-2] % np.pi
                random_theta_sigma = np.sqrt(np.diag(pcov))[-2]
                # rotate theta for drift rate calculation by 90 deg
                random_sigma_x = p1[1]
                random_sigma_y = p1[2]
                if random_sigma_y > random_sigma_x:
                    random_theta += np.pi / 2.
                thetas[dm_trial * mc_trials + mc_trial] = random_theta
                theta_sigmas[dm_trial * mc_trials + mc_trial] = \
                    random_theta_sigma
                drift_rates[dm_trial * mc_trials + mc_trial] = 1.0 / np.tan(
                    -random_theta
                )
            except:
                thetas[dm_trial * mc_trials + mc_trial] = np.nan
                theta_sigmas[dm_trial * mc_trials + mc_trial] = np.nan
                drift_rates[dm_trial * mc_trials + mc_trial] = np.nan
                continue

            if plot_all:
                plt.figure()
                plt.imshow(
                    random_data, aspect="auto", interpolation="none",
                    origin="lower"
                )
                plt.show()

    # apparently it is necessary to do this again
    thetas = thetas % np.pi

    # address wrapping around
    bigdiff = np.nanmax(np.diff(thetas))
    thetas[thetas > bigdiff / 2] -= bigdiff

    np.savez(
        fdir + "ac_drift_rates_{}".format(eventid),
        dms=dms,
        theta=theta,
        theta_sigma=theta_sigma,
        dfdt_data=dfdt_data,
        mc_thetas=thetas,
        mc_theta_sigmas=theta_sigmas,
        mc_drift_rates=drift_rates,
    )

    if plot_result:

        nanmask = np.isnan(thetas)

        plt.figure(figsize = (6.4, 4.8))
        plt.hist(
            thetas[~nanmask],
            color="tab:gray",
            alpha=0.5,
            zorder=1,
            #bins=max(10, int(np.ceil(np.sqrt(np.sum(~nanmask))))),
            bins="auto",
            label="{}x{}({}) trials".format(
                dm_trials, mc_trials, np.sum(~nanmask)),
        )

        ylim = plt.ylim()

        if dfdt_data > 400.0 / (ds.dt_s * 1e3):
            print_dfdt_data = np.inf
        elif dfdt_data < -400.0 / (ds.dt_s * 1e3):
            print_dfdt_data = -np.inf
        else:
            print_dfdt_data = dfdt_data
        plt.axvline(theta, lw=1, color="tab:green", zorder=2)
        plt.text(
            theta,
            ylim[1] / 2.0,
            "$\mu_\mathrm{data}$ = "
            + "{:.3f}".format(theta)
            + " $\Rightarrow$ "
            + "{:+.2f} MHz/ms".format(print_dfdt_data),
            ha="right",
            va="center",
            rotation=90,
            color="tab:green",
            zorder=2,
        )

        mu_theta = np.nanmean(thetas)
        dfdt_mc = 1.0 / np.tan(-mu_theta)
        if dfdt_mc > 400.0 / (ds.dt_s * 1e3):
            print_dfdt_mc = np.inf
        elif dfdt_mc < -400.0 / (ds.dt_s * 1e3):
            print_dfdt_mc = -np.inf
        else:
            print_dfdt_mc = dfdt_mc
        plt.axvline(mu_theta, lw=1, color="tab:blue", zorder=2)
        plt.text(
            mu_theta,
            ylim[1] / 2.0,
            "$\mu_\mathrm{MC}$ = "
            + "{:.3f}".format(mu_theta)
            + " $\Rightarrow$ "
            + "{:+.2f} MHz/ms".format(print_dfdt_mc),
            ha="right",
            va="center",
            rotation=90,
            color="tab:blue",
            zorder=2,
        )

        theta_mc_low = np.nanpercentile(
            thetas, (100.0 - uncertainty_confidence) / 2.0
        )
        dfdt_mc_low = 1.0 / np.tan(-theta_mc_low)
        if dfdt_mc_low > 400.0 / (ds.dt_s * 1e3):
            print_dfdt_mc_low = np.inf
        elif dfdt_mc_low < -400.0 / (ds.dt_s * 1e3):
            print_dfdt_mc_low = -np.inf
        else:
            print_dfdt_mc_low = dfdt_mc_low
        plt.axvline(theta_mc_low, lw=1, ls=":", color="tab:blue", zorder=2)
        plt.text(
            theta_mc_low,
            ylim[1] / 2.0,
            r"{:.1f}% = {:.3f} ".format(
                (100.0 - uncertainty_confidence) / 2.0, theta_mc_low
            )
            + "$\Rightarrow$ "
            + "{:+.2f} MHz/ms".format(print_dfdt_mc_low),
            ha="right",
            va="center",
            rotation=90,
            color="tab:blue",
            zorder=2,
        )

        theta_mc_high = np.nanpercentile(
            thetas, 100.0 - (100.0 - uncertainty_confidence) / 2.0
        )
        dfdt_mc_high = 1.0 / np.tan(-theta_mc_high)
        if dfdt_mc_high > 400.0 / (ds.dt_s * 1e3):
            print_dfdt_mc_high = np.inf
        elif dfdt_mc_high < -400.0 / (ds.dt_s * 1e3):
            print_dfdt_mc_high = -np.inf
        else:
            print_dfdt_mc_high = dfdt_mc_high
        plt.axvline(theta_mc_high, lw=1, ls=":", color="tab:blue", zorder=2)
        plt.text(
            theta_mc_high,
            ylim[1] / 2.0,
            r"{:.1f}% = {:.3f} ".format(
                100.0 - (100.0 - uncertainty_confidence) / 2.0, theta_mc_high
            )
            + "$\Rightarrow$ "
            + "{:+.2f} MHz/ms".format(print_dfdt_mc_high),
            ha="right",
            va="center",
            rotation=90,
            color="tab:blue",
            zorder=2,
        )

        # show only containment region set by `detection_confidence`
        xlim_low = np.nanpercentile(
            thetas, (100.0 - detection_confidence) / 2.0
        )
        xlim_high = np.nanpercentile(
            thetas, 100.0 - (100.0 - detection_confidence) / 2.0
        )
        plt.xlim(xlim_low, xlim_high)

        # check if angles are in the same quadrant of the unit circle
        if int(np.ceil(2 * xlim_low / np.pi)) \
            == int(np.ceil(2 * xlim_high / np.pi)):
            plt.title(
                "{} {} ".format(source, eventid)
                + r"$\Delta$DM = "
                + "{:.1f} pc/cc ".format(dm_uncertainty)
                + "df/dt ({:.1f}%) = {:+.2f}".format(
                    uncertainty_confidence, dfdt_data)
                + r"$_{{-{:.2f}}}^{{+{:.2f}}}$ MHz/ms".format(
                    abs(dfdt_data - dfdt_mc_low), abs(dfdt_data - dfdt_mc_high)
                )
            )

            print(
                "{} -- {} -- df/dt ".format(source, eventid)
                + "({:.1f}%) = {:+.2f}-{:.2f}+{:.2f} MHz/ms".format(
                    uncertainty_confidence,
                    dfdt_data,
                    abs(dfdt_data - dfdt_mc_low),
                    abs(dfdt_data - dfdt_mc_high),
                )
            )

            constrained = True
        else:
            plt.title(
                "{} {} ".format(source, eventid)
                + r"$\Delta$DM = "
                + "{:.1f} pc/cc ".format(dm_uncertainty)
                + "df/dt unconstrained ({:.1f}%)".format(detection_confidence)
            )

            print(
                "{} -- {} -- df/dt unconstrained ({:.1f}%)".format(
                    source, eventid, detection_confidence
                )
            )

            constrained = False

        plt.xlabel(r"$\theta$ (rad)")
        plt.ylabel("# of MC realizations")
        plt.legend(frameon=False)

        plt.tight_layout()
        plt.savefig(
            fdir + "ac_mc_drift_{}.png".format(eventid), dpi=80,
            bbox_inches="tight"
        )
        plt.close()

    return constrained, dfdt_data, dfdt_mc, dfdt_mc_low, dfdt_mc_high

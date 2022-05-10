""" Utilties to interface gwsurrogate models with the generic PyCBC
gw waveform interfaces
"""
import gwsurrogate as gws

_cached_models = {}

def gws_td_gen(**params):
    """ Generate pycbc format time domain GW polarizations

    Parameters
    ----------
    params: dict
        A dictionary containing the waveform parameters

    Returns
    -------
    hp: pycbc.types.TimeSeries
        The time series containg the plus polarization in the
        radiation frame.
    hc: pycbc.types.TimeSeries
        The time series containg the cross polarization in the
        radiation frame.
    """
    import gwsurrogate as gws
    from pycbc.types import TimeSeries
    from pycbc import conversions as conv

    # Parameter names should follow the convention of
    # https://github.com/gwastro/pycbc/blob/master/pycbc/waveform/parameters.py#L162
    # but are not limited to these names for additional effects
    apx = params['approximant'].replace('GWS-', '')

    if apx not in _cached_models:
        _cached_models[apx] = gws.LoadSurrogate(apx)
    model = _cached_models[apx]

    # convert from pycbc parameter name conventions to gwsurrogate
    q1 = params['mass1'] / params['mass2']
    q2 = params['mass2'] / params['mass1']


    q = [q1, q2]
    chi = [[params['spin1x'], params['spin1y'], params['spin1z']],
           [params['spin2x'], params['spin2y'], params['spin2z']]]

    # sort so first mass is larger
    if q1 >= 1:
        a, b = 0, 1
    else:
        a, b = 1, 0

    tides = None
    if model.keywords['Tidal']:
        lam = [params['lambda1'], params['lambda2']]
        tides = {'Lambda1':lam[a], 'Lambda2':lam[b]}
        tides = {k:tides[k] if tides[k] is not None else 0.0 for k in tides}

    M = params['mass1'] + params['mass2']
    if params['f_ref'] == 0:
        params['f_ref'] = params['f_lower']

    _, h, _ = model(q[a], chi[a], chi[b],
                    M=M,
                    dist_mpc=params['distance'],
                    dt=params['delta_t'],
                    phi_ref=params['coa_phase'],
                    f_ref=params['f_ref'],
                    f_low=params['f_lower'],
                    tidal_opts=tides,
                    inclination=params['inclination'],
                    units='mks')

    # Define time=0 as the waveform peak to be consistent with
    # convention
    peak_time = abs(h).argmax() * params['delta_t']
    hp = TimeSeries(h.real, delta_t=params['delta_t'], epoch=-peak_time)
    hc = TimeSeries(h.imag, delta_t=params['delta_t'], epoch=-peak_time)
    return hp, hc

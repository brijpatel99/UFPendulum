"""
config/pendulum_params.py

Pendulum physical parameters and derived quantities as ao scalars.

Usage:
    # Default parameters
    pp = PendulumParams()

    # Override one or more base parameters
    pp = PendulumParams(T0=2500, arm=0.200)

    # Change a parameter after init and recompute derived quantities
    pp.T0 = ao(2500, 's')
    pp.recompute()

NOTE: Values here reflect the most current known configuration.
      For critical results, hard-code values directly in your script.
"""

import logging
import numpy as np
from ltpda.ao import ao
from ltpda import CData, TSData, FSData, XYData
from ltpda.utils.unit import Unit

logger = logging.getLogger(__name__)

# Physical constants
KB          = 1.38e-23          # Boltzmann constant (m^2 kg s^-2 K^-1)
TEMP        = 293.0             # Room temperature (K)
C           = 299792458         # Speed of light (m s^-1)
H_PLANCK    = 6.62607015e-34    # Plancks constant (m^2 kg s^-1)

# Frequency axis for spectral quantities: 1e-6 to 100 Hz
_F_VALS = np.logspace(-6, 2, num=801)


class PendulumParams:
    """
    Common pendulum parameters and derived quantities as ao scalars.

    Base parameters can be overridden at construction or afterwards via
    direct attribute assignment followed by recompute().

    Base parameters Pendulum geometry/dynamics
    -------------------------------------------
    II      : moment of inertia          (kg m^2)
    Q       : quality factor             (dimensionless)
    T0      : natural torsional period   (s)
    arm     : pendulum arm length        (m)
    d       : test mass gap              (m)
    ncap    : capacitive readout noise   (m/rHz)
    nifo    : IFO readout noise          (m/rHz)
    LISAr   : LISA acceleration req.     (m s^-2 /rHz)

    Capacitance gradients
    ---------------------
    LISAdcdxmeas  : measured dC/dx for LISA-like TM   (F/m)
    LISAdcdxpp    : peak-peak dC/dx                   (F/m)
    LISAd2cdx2pp  : peak-peak d2C/dx2                 (F/m^2)
    Sdcdxpp       : dC/dx for Simple TM               (F/m)
    Sd2cdx2pp     : d2C/dx2 for Simple TM             (F/m^2)

    COMSOL capacitance estimates
    ----------------------------
    C_phi_LISA    : X elec to TM C_phi for LISA-like      (F)
    C_phi_p_LISA  : X elec to TM dC/dphi for LISA-like    (F/rad)
    C_phi_pp_LISA : X elec to TM d2C/d2phi for LISA-like  (F/rad^2)

    C_h_phi_LISA    : X elec to EH C_h_phi for LISA-like      (F)
    C_h_phi_p_LISA  : X elec to EH dC_h/dphi for LISA-like    (F/rad)
    C_h_phi_pp_LISA : X elec to EH d2C_h/d2phi for LISA-like  (F/rad^2)

    Derived quantities (recomputed on init and after recompute())
    -------------------------------------------------------------
    w0      : natural angular frequency  (rad/s)
    gamma   : torsional spring constant  (kg m^2 s^-2 rad^-1)
    beta    : dissipative constant       (kg m^2 s^-1 rad^-1)
    H       : torque → angle transfer function ao (fsdata)
    NtoLISA : torque to LISA accel. conversion factor
    SthN    : fiber thermal noise ASD    (kg m^2 s^-2 /rHz)
    ScapN   : capacitive readout noise ASD referred to torque
    SifoN   : IFO readout noise ASD referred to torque
    SLISAa  : LISA acceleration noise requirement ASD
    SLISAf  : LISA force noise requirement ASD

    Default plist equivalents (as dicts for use as **kwargs)
    ---------------------------------------------------------
    psd_kwargs  : default kwargs for ASD plots
    lpsd_kwargs : default kwargs for LPSD ASD plots
    filt_kwargs : default kwargs for filters
    """

    # ------------------------------------------------------------------ #
    # Base parameter defaults and computed (from base)
    # ------------------------------------------------------------------ #
    _DEFAULTS = {
        # Pendulum Geometry/Dynamics params
        'II':              (1.65e-2,   'kg m^2'),
        'Q':               (964,       ''),
        'T0':              (3000,      's'),
        'arm':             (0.222,     'm'),
        'd':               (4e-3,      'm'),
        'ncap':            (60e-9,     'm Hz^(-1/2)'),
        'nifo':            (0.6e-9,    'm Hz^(-1/2)'),
        'LISAr':           (3e-15,     'm s^(-2) Hz^(-1/2)'),
        # Capacitance gradients
        'LISAdcdxmeas':    (2.123e-10,   'F m^-1'),
        'LISAdcdxpp':      (2.887e-10,   'F m^-1'),
        'LISAd2cdx2pp':    (1.444e-7,    'F m^-2'),
        'Sdcdxpp':         (1.155e-10,   'F m^-1'),
        'Sd2cdx2pp':       (3.609375e-6, 'F m^-2'),
        # COMSOL
        'C_phi_LISA':      (-1.274e-12,  'F'),
        'C_phi_p_LISA':    (84.24e-12,   'F rad^-1'),
        'C_phi_pp_LISA':   (10032.12e-12,'F rad^-2'),
        'C_h_phi_LISA':    (-9.47e-12,   'F'),
        'C_h_phi_p_LISA':  (16.24e-12,   'F rad^-1'),
        'C_h_phi_pp_LISA': (806.61e-12,  'F rad^-2'),
        'C_tot_LISA':      (36.35e-12,   'F'),
    }

    _COMPUTED = [
        # Pendulum Dynamics 
        'w0',       # Angular freq of pendulum
        'gamma',    # Torsional spring constant: gamma = w0^2 * II  [kg m^2 s^-2 rad^-1]
        'beta',     # Dissipative constant: beta = II * w0 / Q  [kg m^2 s^-1 rad^-1]
    #    'H',        # Transfer function: torque → angle
        # Noise
        'NtoLISA',   # Torque to LISA acceleration conversion
        'SthN',      # Thermal noise ASD: sqrt(4 * kb * T * gamma / (2pi * f * Q))
    #    'ScapN',     # Capacitive readout noise referred to torque
    #    'SifoN',     # IFO readout noise referred to torque
        'SLISAa',    # LISA acceleration noise ASD (L3 proposal 2017 form)
        'SLISAf',    # LISA force noise ASD
        # Misc
#        'f',        # Frequency axis ao (fsdata)
        # 'kb',       # Boltzmann const [m^2 kg s^-2 K^-1]
        # 'T',        # Temperature [K]
    ]

    def __init__(self, **overrides):
        """
        Initialise pendulum parameters with optional overrides.

        Each override can be either:
        - a plain float/int (assumed same units as default)
        - an ao

        Example:
            pp = PendulumParams(T0=2500, arm=0.200)
            pp = PendulumParams(T0=ao(2500, 's'))
        """

        self._set_base_params(overrides)
        self._compute_derived()

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def recompute(self):
        """
        Recompute all derived quantities from current base parameters.
        Call after changing any base parameter directly:

            pp.Q = ao(3000)
            pp.recompute()
        """
        self._compute_derived()
        logger.debug("PendulumParams: derived quantities recomputed")

    def __repr__(self):
        lines = ["PendulumParams:"]
        lines.append("Base Params:")
        for name, (val, units) in self._DEFAULTS.items():
            current = getattr(self, name)
            y = current.ydata()[0] if hasattr(current.ydata(), '__len__') else current.ydata()
            lines.append("  %-20s = %g %s" % (name, y, units))

        lines.append("Computed Params:")
        for name in self._COMPUTED:
            current = getattr(self, name)
            y = current.ydata()[0] if hasattr(current.ydata(), '__len__') else current.ydata()
            lines.append("  %-20s = %g %s" % (name, y, current.yunits()))
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Default plist equivalents
    # ------------------------------------------------------------------ #

    @property
    def psd_kwargs(self) -> dict:
        """Default kwargs for ASD plots."""
        return {'scale': 'asd'}

    @property
    def lpsd_kwargs(self) -> dict:
        """Default kwargs for LPSD ASD plots."""
        return {'scale': 'asd', 'kdes': 5, 'lmin': 0, 'jdes': 1000}

    @property
    def filt_kwargs(self) -> dict:
        """Default kwargs for filters."""
        return {'fc': 0.01, 'order': 2, 'method': 'filtfilt'}

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _set_base_params(self, overrides: dict):
        """Set base parameters from defaults, applying any overrides."""
        for name, (default_val, units) in self._DEFAULTS.items():
            if name in overrides:
                val = overrides[name]
                if isinstance(val, (CData, TSData, FSData, XYData)):
                    setattr(self, name, val)
                else:
                    setattr(self, name, ao(vals=float(val), yunits=units))
                logger.debug("PendulumParams: override %s = %s", name, val)
            else:
                setattr(self, name, ao(vals=float(default_val), yunits=units))

        # Warn on unrecognised overrides
        for name in overrides:
            if name not in self._DEFAULTS:
                logger.warning("PendulumParams: unknown parameter '%s' — ignoring", name)

    def _compute_derived(self):
        """Compute all derived quantities from current base parameters."""
        # Frequency axis ao (fsdata)
        f = ao(xvals=_F_VALS, yvals=_F_VALS, type='fsdata', yunits='Hz')

        kb = ao(KB,   yunits='m^2 kg s^-2 K^-1')
        T  = ao(TEMP, yunits='K')
        # c  = ao(C, yunits='m s^-1')
        # h  = ao(H_PLANCK, yunits='m^2 kg s^-1')
        # lam_UV = ao(250e-9, yunits='m')
        # nu_UV = ao(c/lam_UV, yunits='s^-1')
        # E_UV = ao(h*nu_UV / lam_UV, yunits='eV')

        # Angular frequency
        self.w0 = (2 * np.pi / self.T0) * ao(1, yunits='rad')
        self.w0.setName('w0')

        # Torsional spring constant: gamma = w0^2 * II  [kg m^2 s^-2 rad^-1]
        self.gamma = (self.w0 ** 2) * self.II * ao(1, yunits='rad^(-3)')
        self.gamma.setYunits(self.gamma.yunits().simplify())
        self.gamma.setName('Gamma')

        # Dissipative constant: beta = II * w0 / Q  [kg m^2 s^-1 rad^-1]
        self.beta = self.II * self.w0 / self.Q * ao(1, yunits='rad^(-1)')
        self.beta.yaxis.units = self.beta.yunits().simplify()
        self.beta.setName('Beta')

        # Torque to LISA acceleration conversion
        self.NtoLISA = ao(1) / self.arm / ao(1.9, yunits='kg')
        self.NtoLISA.setName('N to LISA acc')

        # Transfer function: torque → angle
        # H(f) = 1 / (gamma * (1 - (2pi*f/w0)^2 + i/Q))
        # self.H = ao(1) / (self.gamma * (ao(1) - (2 * np.pi * f / self.w0 * ao(1, yunits='rad')) ** 2 + ao(1j / self.Q.y)))
        # self.H.yaxis.units = self.H.yunits().simplify()
        # self.H.setName('Torque to angle transfer function')

        # Thermal noise ASD: sqrt(4 * kb * T * gamma / (2pi * f * Q))
        self.SthN = (4 * kb * T * self.gamma / (2 * np.pi * f * self.Q)) ** 0.5
        self.SthN.yaxis.units = 'kg m^2 s^(-2) Hz^(-1/2)'
        self.SthN.setName('Fiber thermal noise')

        # Capacitive readout noise referred to torque
        # self.ScapN = ((2 ** 0.5) * self.ncap / (2 * self.arm) / abs(self.H) * ao(1, yunits='rad'))
        # self.ScapN.yaxis.units = self.ScapN.yunits().simplify()
        # self.ScapN.setName('Readout noise (capacitive)')

        # IFO readout noise referred to torque
        # self.SifoN = self.nifo / (2 * self.arm) / abs(self.H) * ao(1, yunits='rad')
        # self.SifoN.yaxis.units = self.SifoN.yunits().simplify()
        # self.SifoN.setName('Readout noise (interferometer)')

        # LISA acceleration noise ASD (L3 proposal 2017 form)
        self.SLISAa = self.LISAr * ((1 + (0.4e-3 / f) ** 2) ** 0.5 * (1 + (f / ao(8e-3, yunits='Hz')) ** 4) ** 0.5)
        self.SLISAa.yaxis.units = 'm s^-2 Hz^(-1/2)'
        self.SLISAa.setName('LISA acceleration noise requirement')

        # LISA force noise ASD
        self.SLISAf = self.SLISAa * ao(2, yunits='kg')
        self.SLISAf.setName('LISA force noise requirement')

        logger.debug("PendulumParams: all derived quantities computed")

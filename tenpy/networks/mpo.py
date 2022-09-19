"""Matrix product operator (MPO).

An MPO is the generalization of an :class:`~tenpy.networks.mps.MPS` to operators. Graphically::

    |      ^        ^        ^
    |      |        |        |
    |  ->- W[0] ->- W[1] ->- W[2] ->- ...
    |      |        |        |
    |      ^        ^        ^

So each 'matrix' has two physical legs ``p, p*`` instead of just one,
i.e. the entries of the 'matrices' are local operators.
Valid boundary conditions of an MPO are the same as for an MPS
(i.e. ``'finite' | 'segment' | 'infinite'``).
(In general, you can view the MPO as an MPS with larger physical space and bring it into
canoncial form. However, unlike for an MPS, this doesn't simplify calculations.
Thus, an MPO has no `form`.)

We use the following label convention for the `W` (where arrows indicate `qconj`)::

    |            p*
    |            ^
    |            |
    |     wL ->- W ->- wR
    |            |
    |            ^
    |            p


If an MPO describes a sum of local terms (e.g. most Hamiltonians),
some bond indices correspond to 'only identities to the left/right'.
We store these indices in `IdL` and `IdR` (if there are such indices).

Similar as for the MPS, a bond index ``i`` is *left* of site `i`,
i.e. between sites ``i-1`` and ``i``.
"""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

import numpy as np
from scipy.linalg import expm
import warnings
import sys

from ..linalg import np_conserved as npc
from .site import group_sites, Site
from ..tools.string import vert_join
from .mps import MPS as _MPS  # only for MPS._valid_bc
from .mps import MPSEnvironment
from .terms import OnsiteTerms, CouplingTerms, MultiCouplingTerms
from ..tools.misc import add_with_None_0
from ..tools.math import lcm
from ..tools.params import asConfig
from ..algorithms.truncation import TruncationError, svd_theta

__all__ = ['MPO', 'make_W_II', 'MPOGraph', 'MPOEnvironment', 'grid_insert_ops']


class MPO:
    """Matrix product operator, finite (MPO) or infinite (iMPO).

    Parameters
    ----------
    sites : list of :class:`~tenpy.models.lattice.Site`
        Defines the local Hilbert space for each site.
    Ws : list of :class:`~tenpy.linalg.np_conserved.Array`
        The matrices of the MPO. Should have labels ``wL, wR, p, p*``.
    bc : {'finite' | 'segment' | 'infinite'}
        Boundary conditions as described in :mod:`~tenpy.networks.mps`.
        ``'finite'`` requires ``Ws[0].get_leg('wL').ind_len = 1``.
    IdL : (iterable of) {int | None}
        Indices on the bonds, which correpond to 'only identities to the left'.
        A single entry holds for all bonds.
    IdR : (iterable of) {int | None}
        Indices on the bonds, which correpond to 'only identities to the right'.
    max_range : int | np.inf | None
        Maximum range of hopping/interactions (in unit of sites) of the MPO. ``None`` for unknown.
    explicit_plus_hc : bool
        If True, this flag indicates that the hermitian conjugate of the MPO should be
        computed and added at runtime, i.e., `self` is not (necessarily) hermitian.

    Attributes
    ----------
    chinfo : :class:`~tenpy.linalg.np_conserved.ChargeInfo`
        The nature of the charge.
    sites : list of :class:`~tenpy.models.lattice.Site`
        Defines the local Hilbert space for each site.
    dtype : type
        The data type of the `_W`.
    bc : {'finite' | 'segment' | 'infinite'}
        Boundary conditions as described in :mod:`~tenpy.networks.mps`.
        ``'finite'`` requires ``Ws[0].get_leg('wL').ind_len = 1``.
    IdL : list of {int | None}
        Indices on the bonds (length `L`+1), which correpond to 'only identities to the left'.
        ``None`` for bonds where it is not set.
        In standard form, this is `0` (except for unset bonds in finite case)
    IdR : list of {int | None}
        Indices on the bonds (length `L`+1), which correpond to 'only identities to the right'.
        ``None`` for bonds where it is not set.
        In standard form, this is the last index on the bond (except for unset bonds in finite case).
    max_range : int | np.inf | None
        Maximum range of hopping/interactions (in unit of sites) of the MPO. ``None`` for unknown.
    grouped : int
        Number of sites grouped together, see :meth:`group_sites`.
    explicit_plus_hc : bool
        If True, this flag indicates that the hermitian conjugate of the MPO should be
        computed and added at runtime, i.e., `self` is not (necessarily) hermitian.
    _W : list of :class:`~tenpy.linalg.np_conserved.Array`
        The matrices of the MPO. Labels are ``'wL', 'wR', 'p', 'p*'``.
    _valid_bc : tuple of str
        Class attribute. Valid boundary conditions; the same as for an MPS.
    """

    _valid_bc = _MPS._valid_bc  # same valid boundary conditions as an MPS.

    def __init__(self,
                 sites,
                 Ws,
                 bc='finite',
                 IdL=None,
                 IdR=None,
                 max_range=None,
                 explicit_plus_hc=False):
        self.sites = list(sites)
        self.chinfo = self.sites[0].leg.chinfo
        self.dtype = dtype = np.find_common_type([W.dtype for W in Ws], [])
        self._W = [W.astype(dtype, copy=True) for W in Ws]
        self.IdL = self._get_Id(IdL, len(sites))
        self.IdR = self._get_Id(IdR, len(sites))
        self.grouped = 1
        self.bc = bc
        self.max_range = max_range
        self.explicit_plus_hc = explicit_plus_hc
        self.test_sanity()

    def save_hdf5(self, hdf5_saver, h5gr, subpath):
        """Export `self` into a HDF5 file.

        This method saves all the data it needs to reconstruct `self` with :meth:`from_hdf5`.

        Specifically, it saves
        :attr:`sites`,
        :attr:`chinfo`,
        :attr:`max_range` (under these names),
        :attr:`_W` as ``"tensors"``,
        :attr:`IdL` as ``"index_identity_left"``,
        :attr:`IdR` as ``"index_identity_right"``, and
        :attr:`bc` as ``"boundary_condition"``.
        Moreover, it saves :attr:`L`, :attr:`explicit_plus_hc` and :attr:`grouped` as HDF5 attributes,
        as well as the maximum of :attr:`chi` under the name :attr:`max_bond_dimension`.

        Parameters
        ----------
        hdf5_saver : :class:`~tenpy.tools.hdf5_io.Hdf5Saver`
            Instance of the saving engine.
        h5gr : :class`Group`
            HDF5 group which is supposed to represent `self`.
        subpath : str
            The `name` of `h5gr` with a ``'/'`` in the end.
        """
        hdf5_saver.save(self.sites, subpath + "sites")
        hdf5_saver.save(self.chinfo, subpath + "chinfo")
        hdf5_saver.save(self._W, subpath + "tensors")
        hdf5_saver.save(self.IdL, subpath + "index_identity_left")
        hdf5_saver.save(self.IdR, subpath + "index_identity_right")
        h5gr.attrs["grouped"] = self.grouped
        hdf5_saver.save(self.bc, subpath + "boundary_condition")
        hdf5_saver.save(self.max_range, subpath + "max_range")
        h5gr.attrs["explicit_plus_hc"] = self.explicit_plus_hc
        h5gr.attrs["L"] = self.L  # not needed for loading, but still usefull metadata
        h5gr.attrs["max_bond_dimension"] = np.max(self.chi)  # same

    @classmethod
    def from_hdf5(cls, hdf5_loader, h5gr, subpath):
        """Load instance from a HDF5 file.

        This method reconstructs a class instance from the data saved with :meth:`save_hdf5`.

        Parameters
        ----------
        hdf5_loader : :class:`~tenpy.tools.hdf5_io.Hdf5Loader`
            Instance of the loading engine.
        h5gr : :class:`Group`
            HDF5 group which is represent the object to be constructed.
        subpath : str
            The `name` of `h5gr` with a ``'/'`` in the end.

        Returns
        -------
        obj : cls
            Newly generated class instance containing the required data.
        """
        obj = cls.__new__(cls)  # create class instance, no __init__() call
        hdf5_loader.memorize_load(h5gr, obj)

        obj.sites = hdf5_loader.load(subpath + "sites")
        obj.chinfo = hdf5_loader.load(subpath + "chinfo")
        obj._W = hdf5_loader.load(subpath + "tensors")
        obj.dtype = np.find_common_type([W.dtype for W in obj._W], [])
        obj.IdL = hdf5_loader.load(subpath + "index_identity_left")
        obj.IdR = hdf5_loader.load(subpath + "index_identity_right")
        obj.grouped = hdf5_loader.get_attr(h5gr, "grouped")
        obj.bc = hdf5_loader.load(subpath + "boundary_condition")
        obj.max_range = hdf5_loader.load(subpath + "max_range")
        obj.explicit_plus_hc = h5gr.attrs.get("explicit_plus_hc", False)
        obj.test_sanity()
        return obj

    @classmethod
    def from_grids(cls,
                   sites,
                   grids,
                   bc='finite',
                   IdL=None,
                   IdR=None,
                   Ws_qtotal=None,
                   legs=None,
                   max_range=None,
                   explicit_plus_hc=False):
        """Initialize an MPO from `grids`.

        Parameters
        ----------
        sites : list of :class:`~tenpy.models.lattice.Site`
            Defines the local Hilbert space for each site.
        grids : list of list of list of entries
            For each site (outer-most list) a matrix-grid (corresponding to ``wL, wR``)
            with entries being or representing (see :func:`grid_insert_ops`) onsite-operators.
        bc : {'finite' | 'segment' | 'infinite'}
            Boundary conditions as described in :mod:`~tenpy.networks.mps`.
        IdL : (iterable of) {int | None}
            Indices on the bonds, which correpond to 'only identities to the left'.
            A single entry holds for all bonds.
        IdR : (iterable of) {int | None}
            Indices on the bonds, which correpond to 'only identities to the right'.
        Ws_qtotal : (list of) total charge
            The `qtotal` to be used for each grid. Defaults to zero charges.
        legs : list of :class:`~tenpy.linalg.charge.LegCharge`
            List of charges for 'wL' legs left of each `W`, L + 1 entries.
            The last entry should be the conjugate of the 'wR' leg,
            i.e. identical to ``legs[0]`` for 'infinite' `bc`.
            By default, determine the charges automatically. This is limited to cases where
            there are no "dangling open ends" in the MPO graph. (The :class:`MPOGraph` can handle
            those cases, though.)
        max_range : int | np.inf | None
            Maximum range of hopping/interactions (in unit of sites) of the MPO.
            ``None`` for unknown.
        explicit_plus_hc : bool
            If True, the Hermitian conjugate of the MPO is computed at runtime,
            rather than saved in the MPO.

        See also
        --------
        grid_insert_ops : used to plug in `entries` of the grid.
        tenpy.linalg.np_conserved.grid_outer : used for final conversion.
        """
        chinfo = sites[0].leg.chinfo
        L = len(sites)
        assert len(grids) == L  # wrong arguments?
        grids = [grid_insert_ops(site, grid) for site, grid in zip(sites, grids)]
        if Ws_qtotal is None:
            Ws_qtotal = [chinfo.make_valid()] * L
        else:
            Ws_qtotal = chinfo.make_valid(Ws_qtotal)
            if Ws_qtotal.ndim == 1:
                Ws_qtotal = [Ws_qtotal] * L
        IdL = cls._get_Id(IdL, L)
        IdR = cls._get_Id(IdR, L)
        if legs is None:
            if bc != 'infinite':
                # ensure that we have only a single entry in the first and last leg
                # i.e. project grids[0][:, :] -> grids[0][IdL[0], :]
                # and         grids[-1][:, :] -> grids[-1][:,IdR[-1], :]
                first_grid = grids[0]
                last_grid = grids[-1]
                if len(first_grid) > 1:
                    grids[0] = [first_grid[IdL[0]]]
                    IdL[0] = 0
                    IdR[0] = None
                if len(last_grid[0]) > 1:
                    grids[-1] = [[row[IdR[-1]]] for row in last_grid]
                    IdR[-1] = 0
                    IdL[-1] = None
                legs = _calc_grid_legs_finite(chinfo, grids, Ws_qtotal, None)
            else:
                legs = _calc_grid_legs_infinite(chinfo, grids, Ws_qtotal, None, IdL[0])
        # now build the `W` from the grid
        assert len(legs) == L + 1
        Ws = []
        for i in range(L):
            W = npc.grid_outer(grids[i], [legs[i], legs[i + 1].conj()], Ws_qtotal[i], ['wL', 'wR'])
            Ws.append(W)
        return cls(sites, Ws, bc, IdL, IdR, max_range, explicit_plus_hc)

    def test_sanity(self):
        """Sanity check, raises ValueErrors, if something is wrong."""
        assert self.L == len(self.sites)
        if self.bc not in self._valid_bc:
            raise ValueError("invalid MPO boundary conditions: " + repr(self.bc))
        for i in range(self.L):
            S = self.sites[i]
            W = self._W[i]
            S.leg.test_equal(W.get_leg('p'))
            S.leg.test_contractible(W.get_leg('p*'))
            if self.bc == 'infinite' or i + 1 < self.L:
                W2 = self.get_W(i + 1)
                W.get_leg('wR').test_contractible(W2.get_leg('wL'))
        if not (len(self.IdL) == len(self.IdR) == self.L + 1):
            raise ValueError("wrong len of `IdL`/`IdR`")

    @property
    def L(self):
        """Number of physical sites; for an iMPO the len of the MPO unit cell."""
        return len(self.sites)

    @property
    def dim(self):
        """List of local physical dimensions."""
        return [site.dim for site in self.sites]

    @property
    def finite(self):
        """Distinguish MPO vs iMPO.

        True for an MPO (``bc='finite', 'segment'``), False for an iMPO (``bc='infinite'``).
        """
        assert (self.bc in self._valid_bc)
        return self.bc != 'infinite'

    @property
    def chi(self):
        """Dimensions of the virtual bonds."""
        return [W.get_leg('wL').ind_len for W in self._W] + [self._W[-1].get_leg('wR').ind_len]

    def get_W(self, i, copy=False):
        """Return `W` at site `i`."""
        i = self._to_valid_index(i)
        if copy:
            return self._W[i].copy()
        return self._W[i]

    def set_W(self, i, W):
        """Set `W` at site `i`."""
        i = self._to_valid_index(i)
        self._W[i] = W

    def get_IdL(self, i):
        """Return index of `IdL` at bond to the *left* of site `i`.

        May be ``None``.
        """
        i = self._to_valid_index(i)
        return self.IdL[i]

    def get_IdR(self, i):
        """Return index of `IdR` at bond to the *right* of site `i`.

        May be ``None``.
        """
        i = self._to_valid_index(i)
        return self.IdR[i + 1]

    def enlarge_mps_unit_cell(self, factor=2):
        """Repeat the unit cell for infinite MPS boundary conditions; in place.

        Parameters
        ----------
        factor : int
            The new number of sites in the unit cell will be increased from `L` to ``factor*L``.
        """
        if int(factor) != factor:
            raise ValueError("`factor` should be integer!")
        if factor <= 1:
            raise ValueError("can't shrink!")
        if self.finite:
            raise ValueError("can't enlarge finite MPO")
        factor = int(factor)
        self.sites = factor * self.sites
        self._W = factor * self._W
        self.IdL = factor * self.IdL[:-1] + [self.IdL[-1]]
        self.IdR = factor * self.IdR[:-1] + [self.IdR[-1]]
        self.test_sanity()

    def group_sites(self, n=2, grouped_sites=None):
        """Modify `self` inplace to group sites.

        Group each `n` sites together using the :class:`~tenpy.networks.site.GroupedSite`.
        This might allow to do TEBD with a Trotter decomposition,
        or help the convergence of DMRG (in case of too long range interactions).

        Parameters
        ----------
        n : int
            Number of sites to be grouped together.
        grouped_sites : None | list of :class:`~tenpy.networks.site.GroupedSite`
            The sites grouped together.
        """
        if grouped_sites is None:
            grouped_sites = group_sites(self.sites, n, charges='same')
        else:
            assert grouped_sites[0].n_sites == n
        if self.max_range is not None:
            min_n = max(min([gs.n_sites for gs in grouped_sites]), 1)
            self.max_range = int(np.ceil(self.max_range / min_n))
        Ws = []
        IdL = []
        IdR = [self.IdR[0]]
        i = 0
        for gs in grouped_sites:
            new_W = self.get_W(i).itranspose(['wL', 'p', 'p*', 'wR'])
            for j in range(1, gs.n_sites):
                W = self.get_W(i + j).itranspose(['wL', 'p', 'p*', 'wR'])
                new_W = npc.tensordot(new_W, W, axes=[-1, 0])
            comb = [list(range(1, 1 + 2 * gs.n_sites, 2)), list(range(2, 2 + 2 * gs.n_sites, 2))]
            new_W = new_W.combine_legs(comb, pipes=[gs.leg, gs.leg.conj()])
            Ws.append(new_W.iset_leg_labels(['wL', 'p', 'p*', 'wR']))
            IdL.append(self.get_IdL(i))
            i += gs.n_sites
            IdR.append(self.get_IdR(i - 1))
        IdL.append(self.IdL[-1])
        self.IdL = IdL
        self.IdR = IdR
        self._W = Ws
        self.sites = grouped_sites
        self.grouped = self.grouped * n

    def sort_legcharges(self):
        """Sort virtual legs by charges. In place.

        The MPO seen as matrix of the ``wL, wR`` legs is usually very sparse. This sparsity is
        captured by the LegCharges for these bonds not being sorted and bunched. This requires a
        tensordot to do more block-multiplications with smaller blocks. This is in general faster
        for large blocks, but might lead to a larger overhead for small blocks. Therefore, this
        function allows to sort the virtual legs by charges.
        """
        new_W = [None] * self.L
        perms = [None] * (self.L + 1)
        for i, w in enumerate(self._W):
            w = w.transpose(['wL', 'wR', 'p', 'p*'])
            p, w = w.sort_legcharge([True, True, False, False], [True, True, False, False])
            if perms[i] is not None:
                assert np.all(p[0] == perms[i])
            perms[i] = p[0]
            perms[i + 1] = p[1]
            new_W[i] = w
        self._W = new_W
        chi = self.chi
        for b, p in enumerate(perms):
            IdL = self.IdL[b]
            if IdL is not None:
                self.IdL[b] = np.nonzero(p == IdL)[0][0]
            IdR = self.IdR[b]
            if IdR is not None:
                IdR = IdR % chi[b]
                self.IdR[b] = np.nonzero(p == IdR)[0][0]
        # done

    def make_U(self, dt, approximation='II'):
        r"""Creates the U_I or U_II propagator.

        Approximations of MPO exponentials following :cite:`zaletel2015`.

        Parameters
        ----------
        dt : float|complex
            The time step per application of the propagator.
            Should be imaginary for real time evolution!
        approximation : ``'I' | 'II'``
            Selects the approximation, :meth:`make_U_I` (``'I'``) or :meth:`make_U_II` (``'II'``).

        Returns
        -------
        U : :class:`~tepy.networks.mpo.MPO`
            The propagator, i.e. approximation :math:`U ~= exp(H*dt)`
        """
        if approximation == 'II':
            return self.make_U_II(dt)
        elif approximation == 'I':
            return self.make_U_I(dt)
        raise ValueError(repr(approximation) + " not implemented")

    def make_U_I(self, dt):
        r"""Creates the :math:`U_I` propagator with `W_I` tensors.

        Parameters
        ----------
        dt : float|complex
            The time step per application of the propagator.
            Should be imaginary for real time evolution!

        Returns
        -------
        UI : :class:`~tenpy.networks.mpo.MPO`
            The propagator, i.e. approximation :math:`U_I ~= exp(H*dt)`
        """
        U = [
            self.get_W(i).astype(np.result_type(dt, self.dtype),
                                 copy=True).itranspose(['wL', 'wR', 'p', 'p*'])
            for i in range(self.L)
        ]

        IdLR = []
        for i in range(0, self.L):  # correct?
            U1 = U[i]
            U2 = U[(i + 1) % self.L]
            IdL = self.IdL[i + 1]
            IdR = self.IdR[i + 1]
            assert IdL is not None and IdR is not None
            U1[:, IdL, :, :] = U1[:, IdL, :, :] + dt * U1[:, IdR, :, :]
            keep = np.ones(U1.shape[1], dtype=bool)
            keep[IdR] = False
            U1.iproject(keep, 1)
            if self.finite and i + 1 == self.L:
                keep = np.ones(U2.shape[0], dtype=bool)
                assert self.IdR[0] is not None
                keep[self.IdR[0]] = False
            U2.iproject(keep, 0)

            if IdL > IdR:
                IdLR.append(IdL - 1)
            else:
                IdLR.append(IdL)

        IdL = self.IdL[0]
        IdR = self.IdR[0]
        assert IdL is not None and IdR is not None
        if IdL > IdR:
            IdLR_0 = IdL - 1
        else:
            IdLR_0 = IdL
        IdLR = [IdLR_0] + IdLR

        return MPO(self.sites, U, self.bc, IdLR, IdLR, np.inf)

    def make_U_II(self, dt):
        r"""Creates the :math:`U_II` propagator.

        Parameters
        ----------
        dt : float|complex
            The time step per application of the propagator. Should be imaginary for real time evolution!

        Returns
        -------
        U_II : :class:`~tenpy.networks.mpo.MPO`
            The propagator, i.e. approximation :math:`UII ~= exp(H*dt)`

        """
        dtype = np.result_type(dt, self.dtype)
        IdL = self.IdL
        IdR = self.IdR

        chinfo = self.chinfo
        trivial = chinfo.make_valid()
        U = []
        for i in range(0, self.L):
            labels = ['wL', 'wR', 'p', 'p*']
            W = self.get_W(i).itranspose(labels)
            assert np.all(W.qtotal == trivial)
            DL, DR, _, _ = W.shape
            Wflat = W.to_ndarray()
            proj_L = np.ones(DL, dtype=np.bool_)
            proj_L[IdL[i]] = False
            proj_L[IdR[i]] = False
            proj_R = np.ones(DR, dtype=np.bool_)
            proj_R[IdL[i + 1]] = False
            proj_R[IdR[i + 1]] = False

            #Extract (A, B, C, D)
            D = Wflat[IdL[i], IdR[i + 1], :, :]
            C = Wflat[IdL[i], proj_R, :, :]
            B = Wflat[proj_L, IdR[i + 1], :, :]
            A = Wflat[proj_L, :, :, :][:, proj_R, :, :]  # numpy indexing requires two steps

            W_II = make_W_II(dt, A, B, C, D)

            leg_L, leg_R, leg_p, leg_pconj = W.legs
            new_leg_L = npc.LegCharge.from_qflat(chinfo, [chinfo.make_valid()], leg_L.qconj)
            new_leg_L = new_leg_L.extend(leg_L.project(proj_L)[2])
            new_leg_R = npc.LegCharge.from_qflat(chinfo, [chinfo.make_valid()], leg_R.qconj)
            new_leg_R = new_leg_R.extend(leg_R.project(proj_R)[2])

            W_II = npc.Array.from_ndarray(
                W_II,
                [new_leg_L, new_leg_R, leg_p, leg_pconj],
                dtype=dtype,
                qtotal=trivial,
                labels=labels,
            )
            # TODO: could sort by charges.
            U.append(W_II)
        Id = [0] * (self.L + 1)
        return MPO(self.sites, U, self.bc, Id, Id)

    def expectation_value(self, psi, tol=1.e-10, max_range=100):
        """Calculate ``<psi|self|psi>/<psi|psi>``.

        For a finite MPS, simply contract the network ``<psi|self|psi>``.
        For an infinite MPS, it assumes that `self` is the a of terms, with :attr:`IdL`
        and :attr:`IdR` defined on each site.  Under this assumption,
        it calculates the expectation value of terms with the left-most non-trivial
        operator inside the MPO unit cell and returns the average value per site.

        Parameters
        ----------
        psi : :class:`~tenpy.networks.mps.MPS`
            State for which the expectation value should be taken.
        tol : float
            Ignored for finite `psi`.
            For infinite MPO containing exponentially decaying long-range terms, stop evaluating
            further terms if the terms in `LP` have norm < `tol`.
        max_range : int
            Ignored for finite `psi`.
            Contract at most ``self.L * max_range`` sites, even if `tol` is not reached.
            In that case, issue a warning.

        Returns
        -------
        exp_val : float/complex
            The expectation value of `self` with respect to the state `psi`.
            For an infinite MPS: the density per site.
        """
        if psi.finite:
            return MPOEnvironment(psi, self, psi).full_contraction(0)
        env = MPOEnvironment(psi, self, psi)
        L = self.L
        LP0 = env.init_LP(0)
        masks_L_no_IdL = []
        masks_R_no_IdRL = []
        for i, W in enumerate(self._W):
            mask_L = np.ones(W.get_leg('wL').ind_len, np.bool_)
            mask_L[self.get_IdL(i)] = False
            masks_L_no_IdL.append(mask_L)
            mask_R = np.ones(W.get_leg('wR').ind_len, np.bool_)
            mask_R[self.get_IdL(i + 1)] = False
            mask_R[self.get_IdR(i)] = False
            masks_R_no_IdRL.append(mask_R)
        # contract first site with theta
        theta = psi.get_theta(0, 1)
        LP = npc.tensordot(LP0, theta, axes=['vR', 'vL'])
        LP = npc.tensordot(LP, self._W[0], axes=[['wR', 'p0'], ['wL', 'p*']])
        LP = npc.tensordot(LP, theta.conj(), axes=[['vR*', 'p'], ['vL*', 'p0*']])

        for i in range(1, max(max_range, 1) * L):
            i0 = i % L
            W = self._W[i0]
            if i >= L:
                # have one full unit cell: don't use further terms starting with IdL
                mask_L = masks_L_no_IdL[i0]
                LP.iproject(mask_L, 'wR')
                W = W.copy()
                W.iproject(mask_L, 'wL')
            B = psi.get_B(i, form='B')
            LP = npc.tensordot(LP, B, axes=['vR', 'vL'])
            LP = npc.tensordot(LP, W, axes=[['wR', 'p'], ['wL', 'p*']])
            LP = npc.tensordot(LP, B.conj(), axes=[['vR*', 'p'], ['vL*', 'p*']])

            if i >= L - 1:
                RP = env.init_RP(i)
                current_value = npc.inner(LP,
                                          RP,
                                          axes=[['vR*', 'wR', 'vR'], ['vL*', 'wL', 'vL']],
                                          do_conj=False)
                LP_converged = LP.copy()
                LP_converged.iproject(masks_R_no_IdRL[i0], 'wR')
                if npc.norm(LP_converged) < tol:
                    break  # no more terms left
        else:  # no break
            msg = "Tolerance {0:.2e} not reached within {1:d} sites".format(tol, max_range)
            warnings.warn(msg, stacklevel=2)
        if self.explicit_plus_hc:
            current_value += np.conj(current_value)
        return np.real_if_close(current_value / L)

    def variance(self, psi, exp_val=None):
        """Calculate ``<psi|self^2|psi> - <psi|self|psi>^2``.

        Works only for finite systems. Ignores the :attr:`~tenpy.networks.mps.MPS.norm` of `psi`.

        .. todo ::
            This is a naive, expensive implementation contracting the full network.
            Try to follow :arXiv:`1711.01104` for a better estimate; would that even work in
            the infinite limit?

        Parameters
        ----------
        psi : :class:`~tenpy.networks.mps.MPS`
            State for which the variance should be taken.
        exp_val : float/complex | None
            The result of ``<psi|self|psi> = self.expectation_value(psi)`` if known;
            otherwise obtained from :meth:`expectation_value`.
            (Set this to 0 to obtain only the part ``<psi|self^2|psi>``.)
        """
        if self.bc != 'finite':
            raise ValueError("works only for finite systems")
        if self.L != psi.L:
            raise ValueError("expect same L")
        if psi._p_label != ['p']:
            raise NotImplementedError("not adjusted for non-standard MPS.")
        if self.explicit_plus_hc:
            raise NotImplementedError("not implemented for explicit_plus_hc flag")
        assert self.L >= 1
        if exp_val is None:
            exp_val = self.expectation_value(psi)

        th = psi.get_theta(0, n=1)
        W = self.get_W(0).take_slice(self.get_IdL(0), 'wL')
        contr = npc.tensordot(th, W.replace_label('wR', 'wR1'), axes=['p0', 'p*'])
        contr = npc.tensordot(contr, W.replace_label('wR', 'wR2'), axes=['p', 'p*'])
        contr = npc.tensordot(th.conj(), contr, axes=[['vL*', 'p0*'], ['vL', 'p']])
        for i in range(1, self.L):
            B = psi.get_B(i, form='B')
            W = self.get_W(i)
            contr = npc.tensordot(contr, B, axes=['vR', 'vL'])
            contr = npc.tensordot(contr,
                                  W.replace_label('wR', 'wR1'),
                                  axes=[['wR1', 'p'], ['wL', 'p*']])
            contr = npc.tensordot(contr,
                                  W.replace_label('wR', 'wR2'),
                                  axes=[['wR2', 'p'], ['wL', 'p*']])
            contr = npc.tensordot(contr, B.conj(), axes=[['vR*', 'p'], ['vL*', 'p*']])
        contr = contr.take_slice([self.get_IdR(self.L - 1)] * 2, ['wR1', 'wR2'])
        contr = npc.trace(contr, 'vR', 'vR*')
        return np.real_if_close(contr - exp_val**2)

    def dagger(self):
        """Return hermition conjugate copy of self."""
        # complex conjugate and transpose everything
        Ws = [w.conj().itranspose(['wL*', 'wR*', 'p', 'p*']) for w in self._W]
        # and now revert conjugation of the wL/wR legs
        # rename labels 'wL*' -> 'wL', 'wR*' -> 'wR'
        for w in Ws:
            w.ireplace_labels(['wL*', 'wR*'], ['wL', 'wR'])
        # flip charges and qconj back
        for i in range(self.L - 1):
            Ws[i].legs[1] = wR = Ws[i].legs[1].flip_charges_qconj()
            Ws[i + 1].legs[0] = wR.conj()
        Ws[-1].legs[1] = wR = Ws[-1].legs[1].flip_charges_qconj()
        if self.finite:
            Ws[0].legs[0] = Ws[0].legs[0].flip_charges_qconj()
        else:
            Ws[0].legs[0] = wR.conj()
        return MPO(self.sites, Ws, self.bc, self.IdL, self.IdR, self.max_range)

    def is_hermitian(self, eps=1.e-10, max_range=None):
        """Check if `self` is a hermitian MPO.

        Shorthand for ``self.is_equal(self.dagger(), eps, max_range)``.
        """
        return self.is_equal(self.dagger(), eps, max_range)

    def is_equal(self, other, eps=1.e-10, max_range=None):
        """Check if `self` and `other` represent the same MPO to precision `eps`.

        To compare them efficiently we view `self` and `other` as MPS and compare the overlaps
        ``abs(<self|self> + <other|other> - 2 Re(<self|other>)) < eps*(<self|self>+<other|other>)``

        Parameters
        ----------
        other : :class:`MPO`
            The MPO to compare to.
        eps : float
            Precision threshold what counts as zero.
        max_range : None | int
            Ignored for finite MPS; for finite MPS we consider only the terms contained in the
            sites with indices ``range(self.L + max_range)``.
            None defaults to :attr:`max_range` (or :attr:`L` in case this is infinite or None).

        Returns
        -------
        equal : bool
            Whether `self` equals `other` to the desired precision.
        """
        if self.finite:
            max_i = self.L
        else:
            if max_range is None:
                if self.max_range is None or self.max_range == np.inf:
                    max_range = self.L
                else:
                    max_range = self.max_range
            max_i = self.L + max_range

        def overlap(A, B):
            """<A|B> on sites 0 to max_i."""
            wA = A.get_W(0).take_slice([A.get_IdL(0)], ['wL']).conj()
            wB = B.get_W(0).take_slice([B.get_IdL(0)], ['wL'])
            trAdB = npc.tensordot(wA, wB, axes=[['p*', 'p'], ['p', 'p*']])  # wR* wR
            i = 0
            for i in range(1, max_i):
                trAdB = npc.tensordot(trAdB, A.get_W(i).conj(), axes=['wR*', 'wL*'])
                trAdB = npc.tensordot(trAdB,
                                      B.get_W(i),
                                      axes=[['wR', 'p*', 'p'], ['wL', 'p', 'p*']])
            trAdB = trAdB.itranspose(['wR*', 'wR'])[A.get_IdR(i), B.get_IdR(i)]
            return trAdB

        self_other = 2. * np.real(overlap(other, self))
        norms = overlap(self, self) + overlap(other, other)
        return abs(norms - self_other) < eps * abs(norms)

    def apply(self, psi, options):
        """Apply `self` to an MPS `psi` and compress `psi` in place.

        Options
        -------
        .. cfg:config :: ApplyMPO
            :include: VariationalApplyMPO, ZipUpApplyMPO

            compression_method : ``'SVD' | 'variational' | 'zip_up'``
                Mandatory.
                Selects the method to be used for compression.
                For the `SVD` compression, `trunc_params` is the only other option used.
            trunc_params : dict
                Truncation parameters as described in :cfg:config:`truncation`.


        Parameters
        ----------
        psi : :class:`~tenpy.networks.mps.MPS`
            The state to which `self` should be applied, in place.
        options : dict
            See above.
        """
        options = asConfig(options, "MPO_apply")
        method = options['compression_method']
        trunc_params = options.subconfig('trunc_params')
        if method == 'SVD':
            self.apply_naively(psi)
            return psi.compress_svd(trunc_params)
        elif method == 'variational':
            from ..algorithms.mps_common import VariationalApplyMPO
            return VariationalApplyMPO(psi, self, options).run()
        elif method == 'zip_up':
            trunc_err = self.apply_zipup(psi, options)
            return trunc_err + psi.compress_svd(trunc_params)
        # TODO: zipup method infinite?
        raise ValueError("Unknown compression method: " + repr(method))

    def apply_naively(self, psi):
        """Applies an MPO to an MPS (in place) naively, without compression.

        This function simply contracts the `W` tensors of the MPO to the `B` tensors of the
        MPS, resulting in an MPS with bond dimension `self.chi * psi.chi`.

        .. warning ::
            This function sets only a wild *guess* for the new singular values.
            You should either compress the MPS or at least call
            :meth:`~tenpy.networks.mps.MPS.canonical_form`.
            If you use :meth:`apply` instead, this will be done automatically.

        Parameters
        ----------
        psi : :class:`~tenpy.networks.mps.MPS`
            The MPS to which `self` should be applied. Modified in place!
        """
        bc = psi.bc
        if bc != self.bc:
            raise ValueError("Boundary conditions of MPS and MPO are not the same")
        if psi.L != self.L:
            raise ValueError("Length of MPS and MPO not the same")
        for i in range(psi.L):
            B = npc.tensordot(psi.get_B(i, 'B'), self.get_W(i), axes=('p', 'p*'))
            if i == 0 and bc == 'finite':
                B = B.take_slice(self.get_IdL(i), 'wL')
                B = B.combine_legs(['wR', 'vR'], qconj=[-1])
                B.ireplace_labels(['(wR.vR)'], ['vR'])
                B.legs[B.get_leg_index('vR')] = B.get_leg('vR').to_LegCharge()
            elif i == psi.L - 1 and bc == 'finite':
                B = B.take_slice(self.get_IdR(i), 'wR')
                B = B.combine_legs(['wL', 'vL'], qconj=[1])
                B.ireplace_labels(['(wL.vL)'], ['vL'])
                B.legs[B.get_leg_index('vL')] = B.get_leg('vL').to_LegCharge()
            else:
                B = B.combine_legs([['wL', 'vL'], ['wR', 'vR']], qconj=[+1, -1])
                B.ireplace_labels(['(wL.vL)', '(wR.vR)'], ['vL', 'vR'])
                B.legs[B.get_leg_index('vL')] = B.get_leg('vL').to_LegCharge()
                B.legs[B.get_leg_index('vR')] = B.get_leg('vR').to_LegCharge()
            psi.set_B(i, B, 'B')

        if bc == 'infinite':
            # calculate (rather arbitrary) guess for S[0] (no we don't like it either)
            weight = np.ones(self.get_W(0).shape[self.get_W(0).get_leg_index('wL')]) * 0.05
            weight[self.get_IdL(0)] = 1
            weight = weight / np.linalg.norm(weight)
            S0 = np.kron(weight, psi.get_SL(0))  # order dictated by '(wL,vL)'
        else:
            S0 = np.ones(psi.get_B(0, None).get_leg('vL').ind_len)
        psi.set_SL(0, S0)
        for i in range(psi.L):
            psi.set_SR(i, np.ones(psi.get_B(i, None).get_leg('vR').ind_len))

    def apply_zipup(self, psi, options):
        """Applies an MPO to an MPS (in place) with the zip-up method.

        Described in Ref. :cite:`stoudenmire2010`.

        The 'W' tensors are contracted to the 'B' tensors with intermediate SVD
        compressions, truncated to bond dimensions `chi_max * m_temp`.

        .. warning ::
            The MPS afterwards is only approximately in canonical form
            (under the assumption that self is close to unity).
            You should either compress the MPS or at least call
            :meth:`~tenpy.networks.mps.MPS.canonical_form`.
            If you use :meth:`apply` instead, this will be done automatically.

        Parameters
        ----------
        psi : :class:`~tenpy.networks.mps.MPS`
            The MPS to which `self` should be applied. Modified in place!
        trunc_params : dict
            Truncation parameters as described in :cfg:config:`truncation`.


        Options
        -------
        .. cfg:config :: ZipUpApplyMPO

            trunc_params : dict
                Truncation parameters as described in :cfg:config:`truncation`.
            m_temp: int
                bond dimension will be truncated to `m_temp * chi_max`
            trunc_weight: float
                reduces cut for Schmidt values to `trunc_weight * svd_min`
        """
        options = asConfig(options, "zip_up")
        m_temp = options.get('m_temp', 2)
        trunc_weight = options.get('trunc_weight', 1.)
        trunc_params = options.subconfig('trunc_params')
        relax_trunc = trunc_params.copy()  # relaxed truncation criteria
        relax_trunc['chi_max'] *= m_temp
        if 'svd_min' in relax_trunc.keys():
            relax_trunc['svd_min'] *= trunc_weight
        trunc_err = TruncationError()
        bc = psi.bc
        if bc != self.bc:
            raise ValueError("Boundary conditions of MPS and MPO are not the same")
        if psi.L != self.L:
            raise ValueError("Length of MPS and MPO not the same")
        if bc != 'finite':
            raise ValueError("Only finite boundary conditions implemented")
        for i in range(psi.L):
            B = npc.tensordot(psi.get_B(i, 'B'), self.get_W(i), axes=('p', 'p*'))
            if i == 0 and bc == 'finite':
                B = B.take_slice(self.get_IdL(i), 'wL')
                B = B.combine_legs([['vL', 'p'], ['wR', 'vR']], qconj=[+1, -1])
                U, S, VH, err, norm_new = svd_theta(B, relax_trunc)
                trunc_err += err
                psi.norm *= norm_new
                U = U.split_legs()
                VH = VH.split_legs()
                VH.iscale_axis(S, 'vL')
                psi.set_SR(i, S)
                psi.set_B(i, U, 'A')
            elif i == psi.L - 1 and bc == 'finite':
                B = npc.tensordot(VH, B, axes=(['wR', 'vR'], ['wL', 'vL']))
                B = B.take_slice(self.get_IdR(i), 'wR')
                B = B.combine_legs(['vL', 'p'], qconj=[-1])
                U, S, VH, err, norm_new = svd_theta(B, relax_trunc)
                trunc_err += err
                psi.norm *= norm_new
                U = U.split_legs()
                psi.set_SR(i, S)
                psi.set_B(i, U, 'A')
            else:
                B = npc.tensordot(VH, B, axes=(['wR', 'vR'], ['wL', 'vL']))
                B = B.combine_legs([['vL', 'p'], ['wR', 'vR']], qconj=[1, -1])
                U, S, VH, err, norm_new = svd_theta(B, relax_trunc)
                trunc_err += err
                psi.norm *= norm_new
                U = U.split_legs()
                VH = VH.split_legs()
                VH.iscale_axis(S, 'vL')
                psi.set_SR(i, S)
                psi.set_B(i, U, 'A')

        return trunc_err

    def get_grouped_mpo(self, blocklen):
        """group each `blocklen` subsequent tensors and  return result as a new MPO.

        .. deprecated :: 0.5.0
            Make a copy and use :meth:`group_sites` instead.
        """
        msg = "Use functions from `tenpy.algorithms.exact_diag.ExactDiag.from_H_mpo` instead"
        warnings.warn(msg, FutureWarning, 2)
        from copy import deepcopy
        groupedMPO = deepcopy(self)
        groupedMPO.group_sites(n=blocklen)
        return (groupedMPO)

    def get_full_hamiltonian(self, maxsize=1e6):
        """extract the full Hamiltonian as a ``d**L``x``d**L`` matrix.

        .. deprecated :: 0.5.0
            Use :meth:`tenpy.algorithms.exact_diag.ExactDiag.from_H_mpo` instead.
        """
        msg = "Use functions from `tenpy.algorithms.exact_diag.ExactDiag.from_H_mpo` instead"
        warnings.warn(msg, FutureWarning, 2)
        if (self.dim[0]**(2 * self.L) > maxsize):
            print('Matrix dimension exceeds maxsize')
            return np.zeros(1)
        singlesitempo = self.get_grouped_mpo(self.L)
        # Note: the trace works only for 'finite' boundary conditions
        # where the legs are trivial - otherwise it would give 0 or even raise an error!
        return npc.trace(singlesitempo.get_W(0), axes=[['wL'], ['wR']])

    def _to_valid_index(self, i):
        """Make sure `i` is a valid index (depending on `self.bc`)."""
        if not self.finite:
            return i % self.L
        if i < 0:
            i += self.L
        if i >= self.L or i < 0:
            raise ValueError("i = {0:d} out of bounds for finite MPO".format(i))
        return i

    @staticmethod
    def _get_Id(Id, L):
        """parse the IdL or IdR argument of __init__"""
        if Id is None:
            return [None] * (L + 1)
        else:
            try:
                Id = list(Id)
                if len(Id) != L + 1:
                    raise ValueError("expected list with L+1={0:d} entries".format(L + 1))
                return Id
            except TypeError:
                return [Id] * (L + 1)

    def __add__(self, other):
        """Return an MPO representing `self + other`.

        Requires both `self` and `other` to be in standard sum form with `IdL` and `IdR` being set.

        This is a naive, block-wise addition without any compression!

        Parameters
        ----------
        other : :class:`MPO`
            MPO to be added to `self`.

        Returns
        -------
        sum_mpo : :class:`MPO`
            The sum `self + other`.
        """
        L = self.L
        assert self.bc == other.bc
        assert other.L == L

        ps = [self._get_block_projections(i) for i in range(L + 1)]
        po = [other._get_block_projections(i) for i in range(L + 1)]

        def block(of, l, r):
            block_, pl, pr = of
            l = pl[l]
            r = pr[r]
            if l is None or r is None:
                return None
            # else
            return block_[l, r]

        # l/r = left/rigth,  s/o = self/other
        Ws = []
        IdL = [None] * (L + 1)
        IdL[0] = 0
        IdR = [None] * (L + 1)
        IdR[-1] = -1
        for i in range(L):
            ws = self._W[i].itranspose(['wL', 'wR', 'p', 'p*'])
            wo = other._W[i].itranspose(['wL', 'wR', 'p', 'p*'])
            s = (ws, ps[i], ps[i + 1])
            o = (wo, po[i], po[i + 1])
            onsite = add_with_None_0(block(s, 0, 2), block(o, 0, 2))

            w_grid = [
                [block(s, 0, 0), block(s, 0, 1), block(o, 0, 1), onsite        ],
                [None,           block(s, 1, 1), None,           block(s, 1, 2)],
                [None,           None,           block(o, 1, 1), block(o, 1, 2)],
                [None,           None,           None,           block(s, 2, 2)]
            ]  # yapf: disable
            w_grid = np.array(w_grid, dtype=object)
            if w_grid[0, 0] is None:
                w_grid[0, 0] = block(o, 0, 0)
            if w_grid[0, 0] is not None:
                IdL[i + 1] = 0
            if w_grid[-1, -1] is None:
                w_grid[-1, -1] = block(o, 2, 2)
            if w_grid[-1, -1] is not None:
                IdR[i] = -1
            # now drop rows and columns which are completely zero
            w_is_None = np.array([[(w is None) for w in w_row] for w_row in w_grid], dtype=bool)
            w_grid = w_grid[np.logical_not(np.all(w_is_None, 1)), :]
            w_grid = w_grid[:, np.logical_not(np.all(w_is_None, 0))]
            Ws.append(npc.grid_concat(w_grid, [0, 1]))
        if self.max_range is not None and other.max_range is not None:
            max_range = max(self.max_range, other.max_range)
        else:
            max_range = None
        return MPO(self.sites, Ws, self.bc, IdL, IdR, max_range)

    def _get_block_projections(self, i):
        """projecteions onto (IdL, other, IdR) on bond `i` in range(0, L+1)"""
        if self.finite:  # allows i = L for finite bc
            if i < self.L:
                length = self._W[i].get_leg('wL').ind_len
            else:
                assert i == self.L
                length = self._W[i - 1].get_leg('wR').ind_len
        else:
            i = i % self.L
            length = self._W[i].get_leg('wL').ind_len
        IdL = self.IdL[i]
        IdR = self.IdR[i]
        proj_other = np.ones(length, np.bool_)
        if IdL is None:
            proj_IdL = None
        else:
            proj_IdL = np.zeros(length, np.bool_)
            proj_IdL[IdL] = True
            proj_other[IdL] = False
        if IdR is None:
            proj_IdR = None
        else:
            proj_IdR = np.zeros(length, np.bool_)
            proj_IdR[IdR] = True
            proj_other[IdR] = False
            assert IdR != IdL
        if length == int(IdL is not None) + int(IdR is not None):
            proj_other = None
        return (proj_IdL, proj_other, proj_IdR)


def make_W_II(t, A, B, C, D):
    r"""W_II approx to exp(t H) from MPO parts (A, B, C, D).

    Get the W_II approximation of :cite:`zaletel2015`.

    In the paper, we have two formal parameter "phi_{r/c}" which satisfies
    :math:`\phi_r^2 = phi_c^2 = 0`.  To implement this, we temporarily extend the virtual Hilbert
    space with two hard-core bosons "br, bl". The components of Eqn (11) can be computed for each
    index of the virtual row/column independently
    The matrix exponential is done in the hard-core extended Hilbert space

    Parameters
    ----------
    t : float
        The time step per application of the propagator.
        Should be imaginary for real time evolution!
    A, B, C, D :  :class:`numpy.ndarray`
        Blocks of the MPO tensor to be exponentiated, as defined in :cite:`zaletel2015`.
        Legs ``'wL', 'wR', 'p', 'p*'``; legs projected to a single IdL/IdR can be dropped.
    """

    tB = t / np.sqrt(np.abs(t))  #spread time step across B, C
    tC = np.sqrt(np.abs(t))
    d = D.shape[0]

    #The virtual size of W is  (1+Nr, 1+Nc)
    Nr = A.shape[0]
    Nc = A.shape[1]
    W = np.zeros((1 + Nr, 1 + Nc, d, d), dtype=np.result_type(D, t))

    Id_ = np.array([[1, 0], [0, 1]])  #2x2 operators in a hard-core boson space
    b = np.array([[0, 0], [1, 0]])

    Id = np.kron(Id_, Id_)  #4x4 operators in the 2x hard core boson space
    Br = np.kron(b, Id_)
    Bc = np.kron(Id_, b)
    Brc = np.kron(b, b)
    for r in range(Nr):  #double loop over row / column of A
        for c in range(Nc):
            #Select relevent part of virtual space and extend by hardcore bosons
            h = np.kron(Brc, A[r, c, :, :]) + np.kron(Br, tB * B[r, :, :]) + np.kron(
                Bc, tC * C[c, :, :]) + t * np.kron(Id, D)
            w = expm(h)  #Exponentiate in the extended Hilbert space
            w = w.reshape((2, 2, d, 2, 2, d))
            w = w[:, :, :, 0, 0, :]
            W[1 + r, 1 +
              c, :, :] = w[1, 1]  #This part now extracts relevant parts according to Eqn 11
            if c == 0:
                W[1 + r, 0] = w[1, 0]
            if r == 0:
                W[0, 1 + c] = w[0, 1]
                if c == 0:
                    W[0, 0] = w[0, 0]
        if Nc == 0:  #technically only need one boson
            h = np.kron(Br, tB * B[r, :, :]) + t * np.kron(Id, D)
            w = expm(h)
            w = w.reshape((2, 2, d, 2, 2, d))
            w = w[:, :, :, 0, 0, :]
            W[1 + r, 0] = w[1, 0]
            if r == 0:
                W[0, 0] = w[0, 0]

    if Nr == 0:
        for c in range(Nc):
            h = np.kron(Bc, tC * C[c, :, :]) + t * np.kron(Id, D)
            w = expm(h)
            w = w.reshape((2, 2, d, 2, 2, d))
            w = w[:, :, :, 0, 0, :]
            W[0, 1 + c] = w[0, 1]
            if c == 0:
                W[0, 0] = w[0, 0]
        if Nc == 0:
            W = expm(t * D)

    return W


class MPOGraph:
    """Representation of an MPO by a graph, based on a 'finite state machine'.

    This representation is used for building H_MPO from the interactions.
    The idea is to view the MPO as a kind of 'finite state machine'.
    The **states** or **keys** of this finite state machine life on the MPO bonds *between* the
    `Ws`. They label the indices of the virtul bonds of the MPOs, i.e., the indices on legs
    ``wL`` and ``wR``. They can be anything hash-able like a ``str``, ``int`` or a tuple of them.

    The **edges** of the graph are the entries ``W[keyL, keyR]``, which itself are onsite operators
    on the local Hilbert space. The indices `keyL` and `keyR` correspond to the legs ``'wL', 'wR'``
    of the MPO. The entry ``W[keyL, keyR]`` connects the state ``keyL`` on bond ``(i-1, i)``
    with the state ``keyR`` on bond ``(i, i+1)``.

    The keys ``'IdR'`` (for 'idenity left') and ``'IdR'`` (for 'identity right') are reserved to
    represent only ``'Id'`` (=identity) operators to the left and right of the bond, respectively.

    .. todo ::
        might be useful to add a "cleanup" function which removes operators cancelling each other
        and/or unused states. Or better use a 'compress' of the MPO?

    Parameters
    ----------
    sites : list of :class:`~tenpy.models.lattice.Site`
        Local sites of the Hilbert space.
    bc : {'finite', 'infinite'}
        MPO boundary conditions.
    max_range : int | np.inf | None
        Maximum range of hopping/interactions (in unit of sites) of the MPO. ``None`` for unknown.

    Attributes
    ----------
    sites : list of :class:`~tenpy.models.lattice.Site`
        Defines the local Hilbert space for each site.
    chinfo : :class:`~tenpy.linalg.np_conserved.ChargeInfo`
        The nature of the charge.
    bc : {'finite', 'infinite'}
        MPO boundary conditions.
    max_range : int | np.inf | None
        Maximum range of hopping/interactions (in unit of sites) of the MPO. ``None`` for unknown.
    states : list of set of keys
        ``states[i]`` gives the possible keys at the virtual bond ``(i-1, i)`` of the MPO.
        `L+1` enries.
    graph : list of dict of dict of list of tuples
        For each site `i` a dictionary ``{keyL: {keyR: [(opname, strength)]}}`` with
        ``keyL in states[i]`` and ``keyR in states[i+1]``.
    _grid_legs : None | list of LegCharge
        The charges for the MPO
    """
    def __init__(self, sites, bc='finite', max_range=None):
        self.sites = list(sites)
        self.chinfo = self.sites[0].leg.chinfo
        self.bc = bc
        self.max_range = max_range
        # empty graph
        self.states = [set() for _ in range(self.L + 1)]
        self.graph = [{} for _ in range(self.L)]
        self._ordered_states = None
        self.test_sanity()

    @classmethod
    def from_terms(cls, terms, sites, bc, insert_all_id=True):
        """Initialize an :class:`MPOGraph` from OnsiteTerms and CouplingTerms.

        Parameters
        ----------
        terms : iterable of ``tenpy.networks.terms.*Terms`` classes
            Entries can be :class:`~tenpy.networks.terms.OnsiteTerms`,
            :class:`~tenpy.networks.terms.CouplingTerms`,
            :class:`~tenpy.networks.terms.MultiCouplingTerms` or
            :class:`~tenpy.networks.terms.ExponentialCouplingTerms`.
            All the entries get added to the new :class:`MPOGraph`.
        sites : list of :class:`~tenpy.networks.site.Site`
            Local sites of the Hilbert space.
        bc : ``'finite' | 'infinite'``
            MPO boundary conditions.
        insert_all_id : bool
            Whether to insert identities such that `IdL` and `IdR` are defined on each bond.
            See :meth:`add_missing_IdL_IdR`.

        Returns
        -------
        graph : :class:`MPOGraph`
            Initialized with the given terms.

        See also
        --------
        from_term_list :
            equivalent for representation by :class:`~tenpy.networks.terms.TermList`.
        """
        max_range = max([t.max_range() for t in terms])
        graph = cls(sites, bc, max_range)
        for term in terms:
            term.add_to_graph(graph)
        graph.add_missing_IdL_IdR(insert_all_id)
        return graph

    @classmethod
    def from_term_list(cls, term_list, sites, bc, insert_all_id=True):
        """Initialize from a list of operator terms and prefactors.

        Parameters
        ----------
        term_list : :class:`~tenpy.networks.mps.TermList`
            Terms to be added to the MPOGraph.
        sites : list of :class:`~tenpy.networks.site.Site`
            Local sites of the Hilbert space.
        bc : ``'finite' | 'infinite'``
            MPO boundary conditions.
        insert_all_id : bool
            Whether to insert identities such that `IdL` and `IdR` are defined on each bond.
            See :meth:`add_missing_IdL_IdR`.

        Returns
        -------
        graph : :class:`MPOGraph`
            Initialized with the given terms.

        See also
        --------
        from_terms : equivalent for other representation of terms.
        """
        ot_ct = term_list.to_OnsiteTerms_CouplingTerms(sites)
        return cls.from_terms(ot_ct, sites, bc, insert_all_id)

    def test_sanity(self):
        """Sanity check, raises ValueErrors, if something is wrong."""
        assert len(self.graph) == self.L
        assert len(self.states) == self.L + 1
        if self.bc not in MPO._valid_bc:
            raise ValueError("invalid MPO boundary conditions: " + repr(self.bc))
        for i, site in enumerate(self.sites):
            if site.leg.chinfo != self.chinfo:
                raise ValueError("invalid ChargeInfo for site {i:d}".format(i=i))
            stL, stR = self.states[i:i + 2]
            # check graph
            gr = self.graph[i]
            for keyL in gr:
                assert keyL in stL
                for keyR in gr[keyL]:
                    assert keyR in stR
                    for opname, strength in gr[keyL][keyR]:
                        assert site.valid_opname(opname)
        # done

    @property
    def L(self):
        """Number of physical sites; for infinite boundaries the length of the unit cell."""
        return len(self.sites)

    def add(self, i, keyL, keyR, opname, strength, check_op=True, skip_existing=False):
        """Insert an edge into the graph.

        Parameters
        ----------
        i : int
            Site index at which the edge of the graph is to be inserted.
        keyL : hashable
            The state at bond (i-1, i) to connect from.
        keyR : hashable
            The state at bond (i, i+1) to connect to.
        opname : str
            Name of the operator.
        strength : str
            Prefactor of the operator to be inserted.
        check_op : bool
            Whether to check that 'opname' exists on the given `site`.
        skip_existing : bool
            If ``True``, skip adding the graph node if it exists (with same keys and `opname`).
        """
        i = i % self.L
        if check_op:
            if not self.sites[i].valid_opname(opname):
                raise ValueError("operator {0!r} not existent on site {1:d}".format(opname, i))
        G = self.graph[i]
        if keyL not in self.states[i]:
            self.states[i].add(keyL)
        if keyR not in self.states[i + 1]:
            self.states[i + 1].add(keyR)
        D = G.setdefault(keyL, {})
        if keyR not in D:
            D[keyR] = [(opname, strength)]
        else:
            entry = D[keyR]
            if not skip_existing or not any([op == opname for op, _ in entry]):
                entry.append((opname, strength))

    def add_string(self, i, j, key, opname='Id', check_op=True, skip_existing=True):
        r"""Insert a bunch of edges for an 'operator string' into the graph.

        Terms like :math:`S^z_i S^z_j` actually stand for
        :math:`S^z_i \otimes \prod_{i < k < j} \mathbb{1}_k \otimes S^z_j`.
        This function adds the :math:`\mathbb{1}` terms to the graph.

        Parameters
        ----------
        i, j: int
            An edge is inserted on all bonds between `i` and `j`, `i < j`.
            `j` can be larger than :attr:`L`, in which case the operators are supposed to act on
            different MPS unit cells.
        key: hashable
            The state at bond (i-1, i) to connect from and on bond (j-1, j) to connect to.
            Also used for the intermediate states.
            No operator is inserted on a site `i < k < j` if ``has_edge(k, key, key)``.
        opname : str
            Name of the operator to be used for the string.
            Useful for the Jordan-Wigner transformation to fermions.
        skip_existing : bool
            Whether existing graph nodes should be skipped.

        Returns
        -------
        label_j : hashable
            The `key` on the left of site j to connect to. Usually the same as the parameter `key`,
            except if ``j - i > self.L``, in which case we use the additional labels ``(key, 1)``,
            ``(key, 2)``, ... to generate couplings over multiple unit cells.
        """
        if j < i:
            raise ValueError("j < i not allowed")
        keyL = keyR = key
        for k in range(i + 1, j):
            if (k - i) % self.L == 0:
                # necessary to extend key because keyL is already in use at this bond
                keyR = keyL + (k, opname, opname)  # same structure as for other standard keys
                # (i, op_i, op_str_right_of_i) e.g. in MultiCouplingTerms.add_to_graph
            k = k % self.L
            if not self.has_edge(k, keyL, keyR):
                self.add(k, keyL, keyR, opname, 1., check_op=check_op, skip_existing=skip_existing)
            keyL = keyR
        return keyL

    def add_missing_IdL_IdR(self, insert_all_id=True):
        """Add missing identity ('Id') edges connecting ``'IdL'->'IdL' and ``'IdR'->'IdR'``.

        This function should be called *after* all other operators have been inserted.

        Parameters
        ----------
        insert_all_id : bool
            If ``True``, insert 'Id' edges on *all* bonds.
            If ``False`` and boundary conditions are finite, only insert
            ``'IdL'->'IdL'`` to the left of the rightmost existing 'IdL' and
            ``'IdR'->'IdR'`` to the right of the leftmost existing 'IdR'.
            The latter avoid "dead ends" in the MPO, but some functions (like `make_WI`) expect
            'IdL'/'IdR' to exist on all bonds.
        """
        if self.bc == 'infinite' or insert_all_id:
            max_IdL = self.L  # add identities for all sites
            min_IdR = 0
        else:
            max_IdL = max([0] + [i for i, s in enumerate(self.states[:-1]) if 'IdL' in s])
            min_IdR = min([self.L] + [i for i, s in enumerate(self.states[:-1]) if 'IdR' in s])
        for k in range(0, max_IdL):
            if not self.has_edge(k, 'IdL', 'IdL'):
                self.add(k, 'IdL', 'IdL', 'Id', 1.)
        for k in range(min_IdR, self.L):
            if not self.has_edge(k, 'IdR', 'IdR'):
                self.add(k, 'IdR', 'IdR', 'Id', 1.)
        # done

    def has_edge(self, i, keyL, keyR):
        """True if there is an edge from `keyL` on bond (i-1, i) to `keyR` on bond (i, i+1)."""
        return keyR in self.graph[i].get(keyL, [])

    def build_MPO(self, Ws_qtotal=None):
        """Build the MPO represented by the graph (`self`).

        Parameters
        ----------
        Ws_qtotal : None | (list of) charges
            The `qtotal` for each of the Ws to be generated, default (``None``) means 0 charge.
            A single qtotal holds for each site.

        Returns
        -------
        mpo : :class:`MPO`
            the MPO which self represents.
        """
        self.test_sanity()
        # pre-work: generate the grid
        self._set_ordered_states()
        grids = self._build_grids()
        IdL = [s.get('IdL', None) for s in self._ordered_states]
        IdR = [s.get('IdR', None) for s in self._ordered_states]
        legs, Ws_qtotal = self._calc_legcharges(Ws_qtotal)
        H = MPO.from_grids(self.sites, grids, self.bc, IdL, IdR, Ws_qtotal, legs, self.max_range)
        return H

    def __repr__(self):
        return "<MPOGraph L={L:d}>".format(L=self.L)

    def __str__(self):
        """string showing the graph for debug output."""
        res = []
        for i in range(self.L):
            G = self.graph[i]
            strs = []
            for keyL in self.states[i]:
                s = [repr(keyL)]
                s.append("-" * len(s[-1]))
                D = G.get(keyL, [])
                for keyR in D:
                    s.append(repr(keyR) + ":")
                    for optuple in D[keyR]:
                        s.append("  " + repr(optuple))
                strs.append("\n".join(s))
            res.append(vert_join(strs, delim='|'))
            res.append('')
        # & states on last MPO bond
        res.append(vert_join([repr(keyR) for keyR in self.states[-1]], delim=' |'))
        return '\n'.join(res)

    def _set_ordered_states(self):
        """Define an ordering of the 'states' on each MPO bond.

        Set ``self._ordered_states`` to a list of dictionaries ``{state: index}``.
        """
        res = self._ordered_states = []
        for s in self.states:
            d = {}
            for i, key in enumerate(sorted(s, key=_mpo_graph_state_order)):
                d[key] = i
            res.append(d)

    def _build_grids(self):
        """translate the graph dictionaries into grids for the `Ws`."""
        states = self._ordered_states
        assert (states is not None)  # make sure that _set_ordered_states was called
        grids = []
        for i in range(self.L):
            stL, stR = states[i:i + 2]
            graph = self.graph[i]  # ``{keyL: {keyR: [(opname, strength)]}}``
            grid = [None] * len(stL)
            for keyL, a in stL.items():
                row = [None] * len(stR)
                for keyR, lst in graph[keyL].items():
                    b = stR[keyR]
                    row[b] = lst
                grid[a] = row
            grids.append(grid)
        return grids

    def _calc_legcharges(self, Ws_qtotal):
        """Obtain charges for the virtual legs of the MPO.

        Should only be called after :meth:`_set_ordered_states`.

        Parameters
        ----------
        Ws_qtotal : None | (list of) charges
            The `qtotal` for each of the Ws to be generated, default (``None``) means 0 charge.
            A single qtotal holds for each site.

        Returns
        -------
        legs : list of :class:`~tenpy.linalg.charge.LegCharge`
            LegCharges to be used on each bond, `L+1` entries.
            Entry `i` contains the 'wL' leg of `W[i]`,
            entry `i+1` needs to be conjugated to be used as `wR` leg of `W[i]`.
        Ws_qtotal :  list of qtotal
            Same as argument, but parsed to a list of L charges defaulting to zeros.
        """
        L = self.L
        states = self._ordered_states
        sites = self.sites
        infinite = (self.bc == 'infinite')
        chinfo = self.chinfo

        if Ws_qtotal is None:
            Ws_qtotal = [chinfo.make_valid()] * L
        else:
            Ws_qtotal = chinfo.make_valid(Ws_qtotal)
            if Ws_qtotal.ndim == 1:
                Ws_qtotal = [Ws_qtotal] * L

        charges = [[None] * len(st) for st in states]
        charges[0][states[0]['IdL']] = chinfo.make_valid(None)  # default charge = 0.
        if infinite:
            charges[-1] = charges[0]  # bond is identical

        def travel_q_LR(i, keyL):
            """Transport charges from left to right through the MPO graph.

            Inspect graph edges on site `i` starting on the left with `keyL` and add charges
            for all connections to the right.
            Recursively transport charges from there."""
            l = states[i][keyL]
            site = sites[i]
            st_r = states[i + 1]
            ch_r = charges[i + 1]
            # charge rule: q_left - q_right + op_qtotal = Ws_qtotal
            qL_Wq = charges[i][l] - Ws_qtotal[i]  # q_left - Ws_qtotal
            edges = self.graph[i][keyL]
            for keyR, ops in edges.items():
                r = st_r[keyR]
                qR = ch_r[r]
                if qR is None:
                    op_qtotal = site.get_op(ops[0][0]).qtotal
                    ch_r[r] = qL_Wq + op_qtotal  # solve chargerule for q_right
                    if infinite or i + 1 < L:
                        travel_q_LR((i + 1) % L, keyR)

        travel_q_LR(0, 'IdL')

        # now we can still have unknown edges in the case of "dead ends" in the MPO graph.

        def travel_q_RL(i, keyL):
            """Transport charges from the right to left through the MPO graph.

            Inspect graph edges on site `i` starting on the left with 'keyL', where
            the charge needs to be determined. If one of them has a charge defined on the right,
            use it to determine the charge for keyL and return True.
            If none of them has the charge defined, return False.
            """
            l = states[i][keyL]
            site = sites[i]
            st_r = states[i + 1]
            ch_r = charges[i + 1]
            # charge rule: q_left - q_right + op_qtotal = Ws_qtotal
            Wq = Ws_qtotal[i]  # q_left - Ws_qtotal
            edges = self.graph[i][keyL]
            for keyR, ops in edges.items():
                r = st_r[keyR]
                qR = ch_r[r]
                if qR is not None:
                    op_qtotal = site.get_op(ops[0][0]).qtotal
                    charges[i][l] = Wq + qR - op_qtotal  # solve chargerule for q_left
                    break
            else:  # no break
                return False
            return True

        if not infinite and any([ch is None for ch in charges[-1]]):
            raise ValueError("can't determine all charges on the very right leg of the MPO!")

        max_checks = sys.getrecursionlimit()  # I don't expect interactions with larger range...
        for _ in range(max_checks):  # recursion limit would be hit in travel_q_LR first!
            repeat = False
            for i in reversed(range(L)):
                ch = charges[i]
                for keyL, l in states[i].items():
                    if ch[l] is None:
                        if not travel_q_RL(i, keyL):
                            repeat = True  # couldn't find it out.
            if not repeat:
                break
        else:  # no break
            raise ValueError("MPOGraph with dead ends: can't determine charges")
        # have all charges determined
        # convert to LegCharges
        legs = []
        for ch in charges:
            ch = chinfo.make_valid(ch)
            leg = npc.LegCharge.from_qflat(chinfo, ch, qconj=+1)
            legs.append(leg)
        if infinite:
            legs[-1] = legs[0]  # identical charges
        return legs, Ws_qtotal


class MPOEnvironment(MPSEnvironment):
    """Stores partial contractions of :math:`<bra|H|ket>` for an MPO `H`.

    The network for a contraction :math:`<bra|H|ket>` of an MPO `H` bewteen two MPS looks like::

        |     .------>-M[0]-->-M[1]-->-M[2]-->- ...  ->--.
        |     |        |       |       |                 |
        |     |        ^       ^       ^                 |
        |     |        |       |       |                 |
        |     LP[0] ->-W[0]-->-W[1]-->-W[2]-->- ...  ->- RP[-1]
        |     |        |       |       |                 |
        |     |        ^       ^       ^                 |
        |     |        |       |       |                 |
        |     .------<-N[0]*-<-N[1]*-<-N[2]*-<- ...  -<--.

    We use the following label convention (where arrows indicate `qconj`)::

        |    .-->- vR           vL ->-.
        |    |                        |
        |    LP->- wR           wL ->-RP
        |    |                        |
        |    .--<- vR*         vL* -<-.

    To avoid recalculations of the whole network e.g. in the DMRG sweeps,
    we store the contractions up to some site index in this class.
    For ``bc='finite','segment'``, the very left and right part ``LP[0]`` and
    ``RP[-1]`` are trivial and don't change in the DMRG algorithm,
    but for iDMRG (``bc='infinite'``) they are also updated
    (by inserting another unit cell to the left/right).

    The MPS `bra` and `ket` have to be in canonical form.
    All the environments are constructed without the singular values on the open bond.
    In other words, we contract left-canonical `A` to the left parts `LP`
    and right-canonical `B` to the right parts `RP`.


    Parameters
    ----------
    bra : :class:`~tenpy.networks.mps.MPS`
        The MPS to project on. Should be given in usual 'ket' form;
        we call `conj()` on the matrices directly.
    H : :class:`~tenpy.networks.mpo.MPO`
        The MPO sandwiched between `bra` and `ket`.
        Should have 'IdL' and 'IdR' set on the first and last bond.
    ket : :class:`~tenpy.networks.mpo.MPS`
        The MPS on which `H` acts. May be identical with `bra`.
    init_LP : ``None`` | :class:`~tenpy.linalg.np_conserved.Array`
        Initial very left part ``LP``. If ``None``, build trivial one with
        :meth`init_LP`.
    init_RP : ``None`` | :class:`~tenpy.linalg.np_conserved.Array`
        Initial very right part ``RP``. If ``None``, build trivial one with
        :meth:`init_RP`.
    age_LP : int
        The number of physical sites involved into the contraction yielding `firstLP`.
    age_RP : int
        The number of physical sites involved into the contraction yielding `lastRP`.

    Attributes
    ----------
    H : :class:`~tenpy.networks.mpo.MPO`
        The MPO sandwiched between `bra` and `ket`.
    """
    def __init__(self, bra, H, ket, init_LP=None, init_RP=None, age_LP=0, age_RP=0):
        if ket is None:
            ket = bra
        if ket is not bra:
            ket._gauge_compatible_vL_vR(bra)  # ensure matching charges
        self.bra = bra
        self.ket = ket
        self.H = H
        self.L = L = lcm(lcm(bra.L, ket.L), H.L)
        self._finite = bra.finite
        self.dtype = np.find_common_type([bra.dtype, ket.dtype, H.dtype], [])
        self._LP = [None] * L
        self._RP = [None] * L
        self._LP_age = [None] * L
        self._RP_age = [None] * L
        if init_LP is None:
            init_LP = self.init_LP(0)
        self.set_LP(0, init_LP, age=age_LP)
        if init_RP is None:
            init_RP = self.init_RP(L - 1)
        self.set_RP(L - 1, init_RP, age=age_RP)
        self.test_sanity()

    def test_sanity(self):
        """Sanity check, raises ValueErrors, if something is wrong."""
        assert (self.bra.finite == self.ket.finite == self.H.finite == self._finite)
        # check that the network is contractable
        for b_s, H_s, k_s in zip(self.bra.sites, self.H.sites, self.ket.sites):
            b_s.leg.test_equal(k_s.leg)
            b_s.leg.test_equal(H_s.leg)
        assert any([LP is not None for LP in self._LP])
        assert any([RP is not None for RP in self._RP])

    def init_LP(self, i):
        """Build initial left part ``LP``.

        Parameters
        ----------
        i : int
            Build ``LP`` left of site `i`.

        Returns
        -------
        init_LP : :class:`~tenpy.linalg.np_conserved.Array`
            Identity contractible with the `vL` leg of ``.ket.get_B(i)``,
            multiplied with a unit vector nonzero in ``H.IdL[i]``,
            with labels ``'vR*', 'wR', 'vR'``.
        """
        init_LP = super().init_LP(i)
        leg_mpo = self.H.get_W(i).get_leg('wL').conj()
        IdL = self.H.get_IdL(i)
        init_LP = init_LP.add_leg(leg_mpo, IdL, axis=1, label='wR')
        return init_LP

    def init_RP(self, i):
        """Build initial right part ``RP`` for an MPS/MPOEnvironment.

        Parameters
        ----------
        i : int
            Build ``RP`` right of site `i`.

        Returns
        -------
        init_RP : :class:`~tenpy.linalg.np_conserved.Array`
            Identity contractible with the `vR` leg of ``self.get_B(i)``,
            multiplied with a unit vector nonzero in ``H.IdR[i]``,
            with labels ``'vL*', 'wL', 'vL'``.
        """
        init_RP = super().init_RP(i)
        leg_mpo = self.H.get_W(i).get_leg('wR').conj()
        IdR = self.H.get_IdR(i)
        init_RP = init_RP.add_leg(leg_mpo, IdR, axis=1, label='wL')
        return init_RP

    def get_LP(self, i, store=True):
        """Calculate LP at given site from nearest available one (including `i`).

        The returned ``LP_i`` corresponds to the following contraction,
        where the M's and the N's are in the 'A' form::

            |     .-------M[0]--- ... --M[i-1]--->-   'vR'
            |     |       |             |
            |     LP[0]---W[0]--- ... --W[i-1]--->-   'wR'
            |     |       |             |
            |     .-------N[0]*-- ... --N[i-1]*--<-   'vR*'


        Parameters
        ----------
        i : int
            The returned `LP` will contain the contraction *strictly* left of site `i`.
        store : bool
            Wheter to store the calculated `LP` in `self` (``True``) or discard them (``False``).

        Returns
        -------
        LP_i : :class:`~tenpy.linalg.np_conserved.Array`
            Contraction of everything left of site `i`,
            with labels ``'vR*', 'wR', 'vR'`` for `bra`, `H`, `ket`.
        """
        # actually same as MPSEnvironment, just updated the labels in the doc string.
        return super().get_LP(i, store)

    def get_RP(self, i, store=True):
        """Calculate RP at given site from nearest available one (including `i`).

        The returned ``RP_i`` corresponds to the following contraction,
        where the M's and the N's are in the 'B' form::

            |     'vL'  ->---M[i+1]-- ... --M[L-1]----.
            |                |              |         |
            |     'wL'  ->---W[i+1]-- ... --W[L-1]----RP[-1]
            |                |              |         |
            |     'vL*' -<---N[i+1]*- ... --N[L-1]*---.

        Parameters
        ----------
        i : int
            The returned `RP` will contain the contraction *strictly* rigth of site `i`.
        store : bool
            Wheter to store the calculated `RP` in `self` (``True``) or discard them (``False``).

        Returns
        -------
        RP_i : :class:`~tenpy.linalg.np_conserved.Array`
            Contraction of everything right of site `i`,
            with labels ``'vL*', 'wL', 'vL'`` for `bra`, `H`, `ket`.
        """
        # actually same as MPSEnvironment, just updated the labels in the doc string.
        return super().get_RP(i, store)

    def full_contraction(self, i0):
        """Calculate the energy by a full contraction of the network.

        The full contraction of the environments gives the value
        ``<bra|H|ket> / (norm(|bra>)*norm(|ket>))``,
        i.e. if `bra` is `ket` and normalized, the total energy.
        For this purpose, this function contracts
        ``get_LP(i0+1, store=False)`` and ``get_RP(i0, store=False)``.

        Parameters
        ----------
        i0 : int
            Site index.
        """
        # same as MPSEnvironment.full_contraction, but also contract 'wL' with 'wR'
        if self.ket.finite and i0 + 1 == self.L:
            # special case to handle `_to_valid_index` correctly:
            # get_LP(L) is not valid for finite b.c, so we use need to calculate it explicitly.
            LP = self.get_LP(i0, store=False)
            LP = self._contract_LP(i0, LP)
        else:
            LP = self.get_LP(i0 + 1, store=False)
        # multiply with `S`: a bit of a hack: use 'private' MPS._scale_axis_B
        S_bra = self.bra.get_SR(i0).conj()
        LP = self.bra._scale_axis_B(LP, S_bra, form_diff=1., axis_B='vR*', cutoff=0.)
        # cutoff is not used for form_diff = 1
        S_ket = self.ket.get_SR(i0)
        LP = self.bra._scale_axis_B(LP, S_ket, form_diff=1., axis_B='vR', cutoff=0.)
        RP = self.get_RP(i0, store=False)
        return npc.inner(LP, RP, axes=[['vR*', 'wR', 'vR'], ['vL*', 'wL', 'vL']], do_conj=False)

    def expectation_value(self, ops, sites=None, axes=None):
        """(doesn't make sense)"""
        raise NotImplementedError("doesn't make sense for an MPOEnvironment")

    def _contract_LP(self, i, LP):
        """Contract LP with the tensors on site `i` to form ``self._LP[i+1]``"""
        # same as MPSEnvironment._contract_LP, but also contract with `H.get_W(i)`
        LP = npc.tensordot(LP, self.ket.get_B(i, form='A'), axes=('vR', 'vL'))
        LP = npc.tensordot(self.H.get_W(i), LP, axes=(['p*', 'wL'], ['p', 'wR']))
        axes = (self.bra._get_p_label('*') + ['vL*'], self.ket._p_label + ['vR*'])
        # for a ususal MPS, axes = (['p*', 'vL*'], ['p', 'vR*'])
        LP = npc.tensordot(self.bra.get_B(i, form='A').conj(), LP, axes=axes)
        return LP  # labels 'vR*', 'wR', 'vR'

    def _contract_RP(self, i, RP):
        """Contract RP with the tensors on site `i` to form ``self._RP[i-1]``"""
        # same as MPSEnvironment._contract_RP, but also contract with `H.get_W(i)`
        RP = npc.tensordot(self.ket.get_B(i, form='B'), RP, axes=('vR', 'vL'))
        RP = npc.tensordot(RP, self.H.get_W(i), axes=(['p', 'wL'], ['p*', 'wR']))
        axes = (self.ket._p_label + ['vL*'], self.ket._get_p_label('*') + ['vR*'])
        # for a ususal MPS, axes = (['p', 'vL*'], ['p*', 'vR*'])
        RP = npc.tensordot(RP, self.bra.get_B(i, form='B').conj(), axes=axes)
        return RP  # labels 'vL', 'wL', 'vL*'


def grid_insert_ops(site, grid):
    """Replaces entries representing operators in a grid of ``W[i]`` with npc.Arrays.

    Parameters
    ----------
    site : :class:`~tenpy.networks.site`
        The site on which the grid acts.
    grid : list of list of `entries`
        Represents a single matrix `W` of an MPO, i.e. the lists correspond to the legs
        ``'vL', 'vR'``, and entries to onsite operators acting on the given `site`.
        `entries` may be ``None``, :class:`~tenpy.linalg.np_conserved.Array`, a single string
        or of the form ``[('opname', strength), ...]``, where ``'opname'`` labels an operator in
        the `site`.

    Returns
    -------
    grid : list of list of {None | :class:`~tenpy.linalg.np_conserved.Array`}
        Copy of `grid` with entries ``[('opname', strength), ...]`` replaced by
        ``sum([strength*site.get_op('opname') for opname, strength in entry])``
        and entries ``'opname'`` replaced by ``site.get_op('opname')``.
    """
    new_grid = [None] * len(grid)
    for i, row in enumerate(grid):
        new_row = new_grid[i] = list(row)
        for j, entry in enumerate(new_row):
            if entry is None or isinstance(entry, npc.Array):
                continue
            if isinstance(entry, str):
                new_row[j] = site.get_op(entry)
            else:
                opname, strength = entry[0]
                res = strength * site.get_op(opname)
                for opname, strength in entry[1:]:
                    res = res + strength * site.get_op(opname)
                new_row[j] = res  # replace entry
                # new_row[j] = sum([strength*site.get_op(opname) for opname, strength in entry])
    return new_grid


def _calc_grid_legs_finite(chinfo, grids, Ws_qtotal, leg0):
    """Calculate LegCharges from `grids` for a finite MPO.

    This is the easier case. We just gauge the very first leg to the left to zeros, then all other
    charges (hopefully) follow from the entries of the grid.
    """
    if leg0 is None:
        if len(grids[0]) != 1:
            raise ValueError("finite MPO with len of first bond != 1")
        q = chinfo.make_valid()
        leg0 = npc.LegCharge.from_qflat(chinfo, [q], qconj=+1)
    legs = [leg0]
    for i, gr in enumerate(grids):
        gr_legs = [legs[-1], None]
        gr_legs = npc.detect_grid_outer_legcharge(gr,
                                                  gr_legs,
                                                  qtotal=Ws_qtotal[i],
                                                  qconj=-1,
                                                  bunch=False)
        legs.append(gr_legs[1].conj())
    return legs


def _calc_grid_legs_infinite(chinfo, grids, Ws_qtotal, leg0, IdL_0):
    """Calculate LegCharges from `grids` for an iMPO.

    Similar like :func:`_calc_grid_legs_finite`, but the hard case.
    Initially, we do not know all charges of the first leg; and they have to
    be consistent with the final leg.

    The way this works: gauge 'IdL' on the very left leg to 0,
    then gradually calculate the charges by going along the edges of the graph
    (maybe also over the iMPO boundary).

    When initializing from an MPO graph directly, use :meth:`MPOGraph._calc_legcharges` directly.
    """
    if leg0 is not None:
        # have charges of first leg: simple case, can use the _calc_grid_legs_finite version.
        legs = _calc_grid_legs_finite(chinfo, grids, Ws_qtotal, leg0)
        legs[-1].test_contractible(legs[0])  # consistent?
        return legs
    L = len(grids)
    chis = [len(g) for g in grids]
    charges = [[None] * chi for chi in chis]
    charges.append(charges[0])  # the *same* list is shared for 0 and -1.

    charges[0][IdL_0] = chinfo.make_valid(None)  # default charge = 0.

    for _ in range(1000 * L):  # I don't expect interactions with larger range than that...
        for i in range(L):
            grid = grids[i]
            QsL, QsR = charges[i:i + 2]
            for vL, row in enumerate(grid):
                qL = QsL[vL]
                if qL is None:
                    continue  # don't know the charge on the left yet
                for vR, op in enumerate(row):
                    if op is None:
                        continue
                    # calculate charge qR from the entry of the grid
                    qR = chinfo.make_valid(qL + op.qtotal - Ws_qtotal[i])
                    if QsR[vR] is None:
                        QsR[vR] = qR
                    elif np.any(QsR[vR] != qR):
                        raise ValueError("incompatible charges while creating the MPO")
        if not any(q is None for Qs in charges for q in Qs):
            break
    else:  # no `break` in the for loop, i.e. we are unable to determine all grid legcharges.
        # this should not happen (if we have no bugs), but who knows ^_^
        # if it happens, there might be unconnected parts in the graph
        raise ValueError("Can't determine LegCharge for the MPO")
    legs = [npc.LegCharge.from_qflat(chinfo, qflat, qconj=+1) for qflat in charges[:-1]]
    legs.append(legs[0])
    return legs


def _mpo_graph_state_order(key):
    """Key-function for sorting they `states` of an MPO Graph.

    For standard TeNPy MPOs we expect keys of the form
    ``'IdL'``, ``'IdR'``, ``(i, op_i, opstr)`` and recursively ``key + (j, op_j, opstr)``,
    (Note that op_j can be opstr if ``j-i >= L``.)

    The goal is to ensure that standard TeNPy MPOs yield an upper-right W for the MPO.
    """
    if isinstance(key, tuple):
        return key
    if isinstance(key, str):
        if key == 'IdL':  # should be first
            return (-2, )
        if key == 'IdR':  # should be last
            return (np.inf, )
    # fallback: compare strings
        return (-1, key)
    return (-1, str(key))

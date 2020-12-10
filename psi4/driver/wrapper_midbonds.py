#
# @BEGIN LICENSE
#
# Psi4: an open-source quantum chemistry software package
#
# Copyright (c) 2007-2019 The Psi4 Developers.
#
# The copyrights for code used from other parties are included in
# the corresponding files.
#
# This file is part of Psi4.
#
# Psi4 is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3.
#
# Psi4 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with Psi4; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# @END LICENSE
#

from psi4 import core

import pprint
import numpy as np


def add_midbonds(**kwargs):
    r"""Adds midbond sites (ghost atoms) to a molecule.

    Parameters
    ----------
    molecule : :ref:`molecule <op_py_molecule>`, optional
        The target molecule, if not the last molecule defined.
    point1 : int_or_string_or_list, optional
        Specification of the first point used to define the midbond.
        An integer is interpreted as an atom (0-indexed).
        The strings "com-1" and "com-2" specify the COM of the first or second fragment.
        A list of integers is interpreted as the COM of those atoms
    point2 : int_or_string_or_list, optional
        Specification of the second point used to define the midbond.
        An integer is interpreted as an atom (0-indexed).
        The strings "com-1" and "com-2" specify the COM of the first or second fragment.
        A list of integers is interpreted as the COM of those atoms

    Returns
    -------
    :py:class:`~psi4.core.Molecule` |w--w| molecule with midbond sites added 
    at specified coordinates

    Examples
    --------
    >>> molecule = add_midbonds(molecule, point1=0, point2=4)

    """
    # If molecule isn't specified, get the active one
    molecule = kwargs.pop('molecule', core.get_active_molecule())

    point1 = kwargs.pop('point1')
    point2 = kwargs.pop('point2')
    print('point1', type(point1))
    print('point2', type(point2))

    nfrag = molecule.nfragments()
    molecule_dict = molecule.to_dict()
    pprint.pprint(molecule_dict)

    # add to these atomic lists
    molecule_dict['elbl'] = np.append(molecule_dict['elbl'], '_dummylabel')
    molecule_dict['elea'] = np.append(molecule_dict['elea'], 4)
    molecule_dict['elem'] = np.append(molecule_dict['elem'], 'He')
    molecule_dict['elez'] = np.append(molecule_dict['elez'], 2)
    molecule_dict['geom'] = np.append(molecule_dict['geom'], np.array([0.0, 0.0, 0.0]))
    molecule_dict['mass'] = np.append(molecule_dict['mass'], 4.00260325)
    molecule_dict['real'] = np.append(molecule_dict['real'], False)

    # if only one fragment, add midbonds to this fragment (CCSD(T) case)
    if nfrag == 1:
        pass 

    # if two fragments, add midbonds to a third ghost fragment (SAPT case)
    elif nfrag == 2:
        molecule_dict['fragment_charges'] = np.append(molecule_dict['fragment_charges'], 0.0)
        molecule_dict['fragment_multiplicities'] = np.append(molecule_dict['fragment_multiplicities'], 1)
        molecule_dict['fragment_separators'] = np.append(molecule_dict['fragment_separators'], molecule.natom())

    # if three or more fragments, add midbonds to last fragment (uncommon SAPT case)
    else:
        pass
    
    pprint.pprint(molecule_dict)
    molecule_dict.pop('elbl') # TODO: debug

    return core.Molecule.from_arrays(**molecule_dict)

